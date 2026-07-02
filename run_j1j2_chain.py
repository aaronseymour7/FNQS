"""
FNQS training pipeline for the 20-site 1D J1-J2 Heisenberg chain,
with gamma = J2/J1 as the coupling-family parameter.

    H(gamma) = J1 * sum_i S_i . S_{i+1}  +  gamma*J1 * sum_i S_i . S_{i+2}

CHANGES vs the original version that OOM'd / diverged to NaN:
  - n_samples is now PER-GAMMA sized so the *total* MC budget across
    R=6 systems is in the same ballpark as the paper's total M (they
    report M as the total batch size split across all R systems, not
    M samples per system - see Table I / Appendix V B of the paper).
    2048/gamma * 6 gammas = 12288 total, vs. the old 16384/gamma *
    6 = 98304 total (6x the paper's largest total budget, on a system
    5x smaller than their N=100 chain).
  - Combined with train_fnqs.py's move to QGTOnTheFly, peak memory no
    longer scales with n_samples * n_params * R, which is what caused
    the OOM kills.
  - diag_shift no longer blindly follows the decay schedule: if the
    previous logged step showed instability (low acceptance, high
    R_hat, or a non-converged CG residual), the shift is held instead
    of decaying further that round. Regularization only relaxes once
    the run is actually behaving.
  - The training loop now reports skipped steps (from the trainer's
    non-finite guard) instead of silently corrupting params.
"""
import os
import pickle
import time

import jax
import numpy as np

from train_fnqs import FNQSTrainer

CKPT_PATH = "fnqs_j1j2_n20_ckpt.pkl"


def diag_shift_schedule(it, n_iters, start=0.05, end=0.02):
    frac = it / max(n_iters - 1, 1)
    return start * (end / start) ** frac


def is_step_healthy(out):
    if out.get("skipped"):
        return False
    if np.any(out["acceptance"] < 0.1):
        return False
    if np.any(out["R_hat"] > 1.1):
        return False
    if out["cg_rel_residual"] > 1e-2:
        return False
    return True


if __name__ == "__main__":
    print("JAX devices:", jax.devices())

    N = 20
    gammas = list(np.linspace(0.0, 0.5, 6))
    R = len(gammas)

    N_SAMPLES_PER_GAMMA = 2048  # total budget = N_SAMPLES_PER_GAMMA * R

    trainer = FNQSTrainer(
        N=N,
        gammas=gammas,
        patch=4,
        d_model=64,
        n_heads=8,
        d_ff=256,
        n_layers=3,
        n_samples=N_SAMPLES_PER_GAMMA,
        n_chains=64,
        n_discard_per_chain=32,
        diag_shift=0.05,
        cg_maxiter=400,
        chunk_size=512,
        max_dtheta_norm=5.0,
        seed=0,
    )

    print(
        f"N={N}, n_params={trainer.n_params}, "
        f"n_samples/gamma={N_SAMPLES_PER_GAMMA}, total_samples={N_SAMPLES_PER_GAMMA * R}, "
        f"n_params/total_samples={trainer.n_params / (N_SAMPLES_PER_GAMMA * R):.2f}, R={R} gammas"
    )
    print(f"gammas = {gammas}")

    n_iters = 3000
    lr = 0.02
    log_every = 20
    ckpt_every = 100

    prev_dtheta_norm = None
    last_shift = 0.05
    n_skipped = 0

    t0 = time.time()
    for it in range(n_iters):
        scheduled_shift = diag_shift_schedule(it, n_iters)
        # Only let the shift decay if the previous logged step looked
        # healthy; otherwise hold (or gently raise) it.
        if prev_dtheta_norm is None:
            shift = scheduled_shift
        elif last_healthy:
            shift = min(scheduled_shift, last_shift)
        else:
            shift = max(last_shift, scheduled_shift) * 1.25

        out = trainer.step(lr=lr, diag_shift=shift)
        last_shift = shift

        if out.get("skipped"):
            n_skipped += 1
            last_healthy = False
            if it % log_every == 0 or n_skipped <= 5:
                elapsed = time.time() - t0
                print(f"[{it:5d}] {elapsed:8.1f}s  diag_shift={shift:.4f}  "
                      f"!! STEP SKIPPED (non-finite forces/dtheta) - params unchanged, "
                      f"CG warm-start reset")
            continue

        last_healthy = is_step_healthy(out)

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
                f"|dtheta| = {out['dtheta_norm']:.3f}   "
                f"(skipped so far: {n_skipped})"
            )

            flags = []
            if np.any(out['acceptance'] < 0.1):
                flags.append("LOW ACCEPTANCE (<10%) - shift will be held/raised next step")
            if np.any(out['R_hat'] > 1.1):
                flags.append("HIGH R_HAT (>1.1) - shift will be held/raised next step")
            if out['cg_rel_residual'] > 1e-2:
                flags.append("CG DID NOT CONVERGE (residual > 1e-2)")
            if prev_dtheta_norm is not None and out['dtheta_norm'] > 5 * prev_dtheta_norm:
                flags.append(f"UPDATE SPIKE - |dtheta| jumped {out['dtheta_norm']/max(prev_dtheta_norm,1e-12):.1f}x "
                              "vs previous logged step (was clipped to max_dtheta_norm)")
            for f in flags:
                print(f"    !! {f}")
            prev_dtheta_norm = out['dtheta_norm']

        if it % ckpt_every == 0 and it > 0:
            with open(CKPT_PATH, "wb") as f:
                pickle.dump({"it": it, "params": trainer.params, "gammas": gammas}, f)

    with open(CKPT_PATH, "wb") as f:
        pickle.dump({"it": n_iters, "params": trainer.params, "gammas": gammas}, f)

    print(f"\nTotal skipped steps: {n_skipped}")
    print("\nFinal per-gamma energies (with error bars):")
    for g, e in zip(trainer.gammas, trainer.energy_per_gamma()):
        print(f"  gamma={g:.3f}: {e}")
