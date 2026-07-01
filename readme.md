# FNQS pipeline: 20-site 1D J1-J2 Heisenberg chain, gamma = J2/J1

## Files

- `model.py` — the transformer ansatz. Takes a spin configuration (N sites)
  plus a scalar coupling `gamma = J2/J1`, splits the configuration into
  patches, concatenates `gamma` onto every patch (the O(1)-coupling
  embedding strategy from the FNQS paper), runs it through a small
  transformer encoder, sum-pools, and outputs a complex log-amplitude
  via two real heads (log|psi| and phase) — numerically equivalent to a
  single complex output layer but more stable to train.

- `train_fnqs.py` — `FNQSTrainer`: implements the paper's generalized
  Stochastic Reconfiguration. At each step it samples configurations
  independently for every gamma in a fixed training grid (using
  Metropolis-Hastings with spin-exchange moves, since this sum-pooling
  architecture is not autoregressive), computes the per-gamma forces F_r
  and quantum geometric tensor S_r, averages them across gamma, and takes
  one natural-gradient step on the network parameters shared by all gamma.

- `run_j1j2_chain.py` — the actual N=20 pipeline you asked for, with
  gamma sampled on a grid spanning [0, 0.5] to cover the known
  Majumdar-Ghosh / dimerization transition at J2/J1 = 0.2411.

## What's been validated in this session

- Hamiltonian construction (colored-edge graph -> `nk.operator.Heisenberg`
  with J1/J2 bonds) checked against exact diagonalization at N=8.
- Single-gamma training (standard VMC+SR) converges toward the exact
  ground energy at N=8.
- The multi-gamma generalized-SR loop (the actual FNQS mechanism) was
  run at N=8 with 3 gamma values sharing one set of weights; energies
  decreased monotonically and landed within 0.3-1.8% of exact
  diagonalization for all three gammas after only 60 steps on a
  deliberately tiny network.
- The full pipeline was confirmed to run without errors at the real
  target size, N=20 (184,756-dimensional Hilbert space in the
  total-Sz=0 sector), for several SR steps.

## What's *not* included: a converged N=20 run

Getting a tight, paper-quality N=20 result needs meaningfully more
compute than fits in this session:

1. **Sample budget.** SR requires `n_samples` comfortably larger than
   `n_params`, or the quantum geometric tensor solve is ill-conditioned
   (netket warns about this explicitly). At N=20 with a reasonably
   expressive network you'll have several thousand parameters, so plan
   for `n_samples` in the 4,000-20,000 range.
2. **Iteration count.** The paper's runs use hundreds to low thousands of
   SR steps. Each step here (dense QGT solve) costs O(n_params^2 *
   n_samples); at N=20 that was ~50-100s/step on CPU with 1024 samples
   in this environment.
3. **Solver.** For n_params in the thousands, replace the dense
   `QGTJacobianDense.to_dense()` solve in `train_fnqs.py` with a
   matrix-free conjugate-gradient solve (`QGTJacobianPyTree` or
   netket's `VMC_SR`/minSR driver), which avoids ever forming an
   n_params x n_params matrix and scales far better.
4. **Hardware.** JAX will use a GPU automatically if one is available
   (`jax.devices()`); this session ran on CPU only.

## How to extend

- **Different gamma range / resolution:** just edit the `gammas` list
  in `run_j1j2_chain.py` — e.g. densify near 0.2411 to resolve the
  transition more sharply, or extend past it to see the dimerized phase.
- **Fidelity susceptibility:** once trained, differentiate `model.apply`
  with respect to `gamma` (autodiff through the same forward pass used
  for sampling) to get the FNQS paper's fidelity susceptibility, which
  will peak near the transition — this doesn't require retraining,
  just a `jax.grad`/`jax.jacfwd` wrapper around the existing model.
- **Adding J3 or more couplings:** extend the `gamma` vector's last
  dimension in `model.py` from 1 to however many O(1) couplings you
  want, and adjust the Hamiltonian construction accordingly (as in the
  paper's 2D J1-J2-J3 experiment).
