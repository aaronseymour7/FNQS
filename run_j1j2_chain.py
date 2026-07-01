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

This has been validated (see accompanying notes) to run correctly at
N=20, and to converge correctly at smaller N (N=8) where exact
diagonalization is available for comparison. Getting tight N=20
convergence needs substantially more compute than a quick CPU check:
- increase n_samples so that n_samples >> n_params (currently the
  binding constraint on solve stability -- see README)
- run several hundred to a few thousand SR steps
- ideally run on GPU (set JAX platform accordingly)
- for large n_params, replace the dense QGTJacobianDense solve in
  train_fnqs.py with netket's QGTJacobianPyTree + CG (matrix-free),
  or netket's newer VMC_SR / minSR driver, to avoid forming an
  n_params x n_params dense matrix every step
"""
import time
import numpy as np
from train_fnqs import FNQSTrainer

if __name__ == "__main__":
    N = 20

    # Coupling grid: dense enough to resolve the dimerization transition
    # near J2/J1 ~ 0.2411 (Majumdar-Ghosh point), spanning gapless -> dimerized.
    gammas = list(np.linspace(0.0, 0.5, 6))

    trainer = FNQSTrainer(
        N=N,
        gammas=gammas,
        patch=4,          # 5 patches of 4 sites each
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        n_samples=4096,   # keep >> n_params for a well-conditioned QGT solve
        n_chains=128,
        n_discard_per_chain=32,
        diag_shift=0.01,
        seed=0,
    )

    n_params = sum(
        x.size for x in __import__("jax").tree_util.tree_leaves(trainer.params)
    )
    print(f"N={N}, n_params={n_params}, n_samples=4096, R={len(gammas)} gammas")
    print(f"gammas = {gammas}")

    n_iters = 2000       # paper-scale run; increase/decrease as compute allows
    lr = 0.02
    log_every = 20

    t0 = time.time()
    for it in range(n_iters):
        E = trainer.step(lr=lr)
        if it % log_every == 0:
            elapsed = time.time() - t0
            print(f"[{it:5d}] {elapsed:8.1f}s  E(gamma) = {np.round(E, 3)}")

    print("\nFinal per-gamma energies (with error bars):")
    for g, e in zip(trainer.gammas, trainer.energy_per_gamma()):
        print(f"  gamma={g:.3f}: {e}")
