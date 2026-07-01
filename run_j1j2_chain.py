"""
FNQS training pipeline for the 20-site 1D J1-J2 Heisenberg chain,
with gamma = J2/J1 as the coupling-family parameter.

    H(gamma) = J1 * sum_i S_i . S_{i+1}  +  gamma*J1 * sum_i S_i . S_{i+2}

Trained on a fixed grid of gamma spanning the region around the
Majumdar-Ghosh / dimerization transition (gamma ~ 0.2411), so the
same network learns the ground state as a function of gamma across
both the gapless (small gamma) and dimerized (larger gamma) regimes.

USAGE:
    python run_j1j2_chain.py

CHANGES FROM THE ORIGINAL (see conversation for rationale):
  - QGT is matrix-free (QGTJacobianPyTree + CG in train_fnqs.py), not a
    dense n_params x n_params solve, so it scales to N=20's ~17k params.
  - n_samples raised to 8192 (was 4096): matrix-free CG doesn't
    technically require n_samples > n_params to run, but a smoke test
    at N=8 showed real instability when the ratio is too low
    (n_samples/n_params ~ 0.2 diverged after training). 8192/17250 ~ 0.47
    is a more defensible starting point - watch the R-hat/variance log
    and raise this if runs are still unstable.
  - diag_shift now follows a decay schedule (0.05 -> 0.005) instead of
    a fixed 0.01, per NetKet's documented typical range and to give
    early training more regularization while the QGT is far from
    converged.
  - Per-gamma energy variance and per-step wall time are logged, not
    just the mean energy, so you can see conditioning problems as
    they happen instead of only at the end.
  - Params are checkpointed to disk periodically so a long run can be
    resumed/inspected without losing progress.
"""
import os
import pickle
import time

import jax
import numpy as np

from train_fnqs import FNQSTrainer

CKPT_PATH = "fnqs_j1j2_n20_ckpt.pkl"


def diag_shift_schedule(it, n_iters, start=0.05, end=0.005):
    """Exponential decay from `start` to `end` over the run."""
    frac = it / max(n_iters - 1, 1)
    return start * (end / start) ** frac


if __name__ == "__main__":
    print("JAX devices:", jax.devices())

    N = 20

    # Coupling grid: dense enough to resolve the dimerization transition
    # near J2/J1 ~ 0.2411 (Majumdar-Ghosh point), spanning gapless -> dimerized.
    gammas = list(np.linspace(0.0, 0.5, 6))

    trainer = FNQSTrainer(
        N=N,
        gammas=gammas,
        patch=4,           # 5 patches of 4 sites each
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        n_samples=8192,    # raised from 4096 - see module docstring
        n_chains=256,      # raised from 128 for better decorrelation near the transition
        n_discard_per_chain=64,
        diag_shift=0.05,   # start of the decay schedule; see diag_shift_schedule
        cg_maxiter=250,
        seed=0,
    )

    print(f"N={N}, n_params={trainer.n_params}, n_samples=8192, "
          f"n_params/n_samples={trainer.n_params/8192:.2f}, R={len(gammas)} gammas")
    print(f"gammas = {gammas}")

    n_iters = 2000       # paper-scale run; increase/decrease as compute allows
    lr = 0.02
    log_every = 20
    ckpt_every = 100

    prev_dtheta_norm = None

    t0 = time.time()
    for it in range(n_iters):
        shift = diag_shift_schedule(it, n_iters)
        out = trainer.step(lr=lr, diag_shift=shift)

        if it % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[{it:5d}] {elapsed:8.1f}s  diag_shift={shift:.4f}\n"
                f"    E        = {np.round(out['energy'], 3)}\n"
                f"    Var      = {np.round(out['energy_var'], 3)}\n"
                f"    err      = {np.round(out['error_of_mean'], 3)}\n"
                f"    R_hat    = {np.round(out['R_hat'], 3)}\n"
                f"    tau_corr = {np.round(out['tau_corr'], 2)}\n"
                f"    accept   = {np.round(out['acceptance'], 3)}\n"
                f"    cg_resid = {out['cg_rel_residual']:.4f}   "
                f"|dtheta| = {out['dtheta_norm']:.3f}"
            )

            # red-flag checks - don't fail the run, just make problems visible
            flags = []
            if np.any(out['acceptance'] < 0.1):
                flags.append("LOW ACCEPTANCE (<10%) - chains may be stuck; "
                              "sampler/gradient estimates unreliable this step")
            if np.any(out['R_hat'] > 1.1):
                flags.append("HIGH R_HAT (>1.1) - chains not equilibrated")
            if out['cg_rel_residual'] > 1e-2:
                flags.append("CG DID NOT CONVERGE (residual > 1e-2) - "
                              "raise cg_maxiter or diag_shift")
            if prev_dtheta_norm is not None and out['dtheta_norm'] > 5 * prev_dtheta_norm:
                flags.append(f"UPDATE SPIKE - |dtheta| jumped {out['dtheta_norm']/prev_dtheta_norm:.1f}x "
                              "vs previous logged step")
            for f in flags:
                print(f"    !! {f}")
            prev_dtheta_norm = out['dtheta_norm']

        if it % ckpt_every == 0 and it > 0:
            with open(CKPT_PATH, "wb") as f:
                pickle.dump({"it": it, "params": trainer.params, "gammas": gammas}, f)

    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"it": n_iters, "params": trainer.params, "gammas": gammas}, f)

    print("\nFinal per-gamma energies (with error bars):")
    for g, e in zip(trainer.gammas, trainer.energy_per_gamma()):
        print(f"  gamma={g:.3f}: {e}")
