# FNQS — Foundation Neural-Network Quantum States

A JAX/NetKet implementation of Foundation Neural-Network Quantum States (FNQS), a variational Monte Carlo framework that trains a single Vision Transformer (ViT) ansatz across multiple Hamiltonians simultaneously. The architecture processes **multimodal inputs** — spin configurations and Hamiltonian coupling constants — enabling generalization across coupling regimes and efficient simulation of disordered quantum systems.

Based on: *Foundation Neural-Network Quantum States as a Unified Ansatz for Multiple Hamiltonians*, [Nature Communications (2025)](https://www.nature.com/articles/s41467-025-62098-x).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Parameters Reference](#parameters-reference)
- [Expected Output](#expected-output)

---

## Overview

Standard Neural-Network Quantum States (NQS) are trained for a fixed Hamiltonian. FNQS extends this by treating the coupling constants themselves as inputs to the network, allowing one model to represent ground states across a family of Hamiltonians. The current implementation targets the **J1–J2 Heisenberg spin chain**:

$$H = J_1 \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j + J_2 \sum_{\langle\langle i,j \rangle\rangle} \mathbf{S}_i \cdot \mathbf{S}_j$$

where $\gamma = J_2 / J_1$ is swept across a user-specified range. The model is trained jointly on all $\gamma$ values using **Stochastic Reconfiguration (SR)** with a shared quantum geometric tensor averaged across Hamiltonians.

---

## Architecture

The ansatz is a ViT adapted for multimodal quantum inputs:

1. **Patching** — the spin configuration $\sigma \in \{-1, +1\}^N$ is split into non-overlapping patches of size `patch`.
2. **Coupling injection** — the coupling ratio $\gamma$ is tiled and appended to each patch, giving tokens of dimension `patch + 1`.
3. **Linear embedding** — tokens are projected to dimension `d_model`.
4. **Transformer encoder** — `n_layers` blocks of multi-head self-attention (`n_heads`) and MLP (`mlp_dim`), with pre-layer normalisation.
5. **Output heads** — sum-pooled representation is split into a log-amplitude head and a phase head, producing a complex log-wavefunction $\log \Psi(\sigma, \gamma)$.

Optimization uses the **natural gradient** (SR) with a $\bar{S}$-matrix formed by averaging the quantum geometric tensors across all training $\gamma$ values, solved via conjugate gradient.

---

## Requirements

```
netket>=3.11,<4.0
jax>=0.4.25
jaxlib>=0.4.25
flax>=0.7.4
numpy>=1.24
```

GPU acceleration is strongly recommended for N ≥ 20. The code runs on CPU for small N but is substantially slower.

---

## Installation

```bash
git clone https://github.com/aaronseymour7/FNQS.git
cd FNQS

# CPU-only JAX
pip install "jax>=0.4.25" "jaxlib>=0.4.25" netket flax numpy

# GPU (CUDA 12)
pip install "jax[cuda12]>=0.4.25" netket flax numpy

# Verify JAX device
python -c "import jax; print(jax.devices())"
```

---

## Usage

### Running the N=20 J2 Sweep

After applying the fixes above:

```bash
python run_j1j2_chain.py
```

This trains the FNQS model on six values of $\gamma \in \{0.0, 0.1, 0.2, 0.3, 0.4, 0.5\}$ simultaneously for 2000 SR steps. Key hyperparameters at the top of `run_j1j2_chain.py`:

```python
N          = 20          # system size (number of sites)
J1         = 1.0         # nearest-neighbour coupling (fixed)
gammas     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # J2/J1 values to sweep
n_iters    = 2000        # SR optimization steps
n_samples  = 8192        # total MC samples per step
lr         = 0.01        # learning rate
diag_shift = 0.01        # SR regularisation (diagonal shift on S-matrix)
```

### Changing the System Size

Set `N` in `run_j1j2_chain.py`. The `patch` size must evenly divide `N`:

| N  | Recommended `patch` | Patches |
|----|---------------------|---------|
| 10 | 2                   | 5       |
| 20 | 4                   | 5       |
| 40 | 4 or 8              | 10 or 5 |
| 100| 5 or 10             | 20 or 10|

### Changing the $\gamma$ Grid

Edit the `gammas` list:

```python
# Fine sweep around the J1-J2 critical point (~0.241)
gammas = [0.15, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35]
```

---

## Repository Structure

```
FNQS/
├── model.py            # ViT ansatz: FNQS_J1J2, EncoderBlock, MultiHeadAttention
├── train_fnqs.py       # FNQSTrainer: SR loop, QGT averaging, CG solver
└── run_j1j2_chain.py   # Entry point: N=20 J1-J2 chain sweep
```

### `model.py`

Defines `FNQS_J1J2`, a Flax `nn.Module` with signature:

```python
FNQS_J1J2(
    d_model  = 64,   # embedding / transformer width
    n_heads  = 4,    # attention heads
    n_layers = 2,    # transformer encoder blocks
    mlp_dim  = 128,  # MLP hidden dim inside each encoder block
    patch    = 2,    # spatial patch size (must divide N)
)
```

Input shape: `(batch, N + 1)` — last column is $\gamma$, first N columns are $\pm 1$ spins.
Output shape: `(batch,)` — complex log-amplitude $\log \Psi(\sigma, \gamma)$.

### `train_fnqs.py`

Defines `FNQSTrainer`, which:

- Wraps the model so each $\gamma$ value gets its own `MCState` with a fixed-$\gamma$ apply function.
- Builds `QGTJacobianPyTree` for each $\gamma$ and averages them into $\bar{S}$.
- Averages forces across $\gamma$ values into $\bar{F}$.
- Solves $\bar{S}\, \delta\theta = \bar{F}$ via conjugate gradient.
- Updates shared parameters and syncs all `MCState` objects.

### `run_j1j2_chain.py`

Entry point that constructs the Hamiltonian for each $\gamma$, builds samplers, instantiates `FNQSTrainer`, and runs the SR loop with periodic logging and checkpointing.

---

## Parameters Reference

| Parameter | Default | Description |
|---|---|---|
| `N` | 20 | Number of lattice sites |
| `J1` | 1.0 | Nearest-neighbour coupling |
| `gammas` | [0.0…0.5] | List of $J_2/J_1$ values trained simultaneously |
| `d_model` | 64 | Transformer embedding dimension |
| `n_heads` | 4 | Number of attention heads (`d_model` must be divisible by `n_heads`) |
| `n_layers` | 2 | Number of encoder blocks |
| `mlp_dim` | 128 | MLP width inside each encoder block |
| `patch` | 4 | Patch size (must divide `N`) |
| `n_samples` | 8192 | Total MCMC samples per SR step |
| `n_chains` | 64 | Number of Markov chains (recommended after fix) |
| `n_discard_per_chain` | 32 | Thermalization steps per chain (recommended after fix) |
| `n_iters` | 2000 | Total SR optimization steps |
| `lr` | 0.01 | Learning rate |
| `diag_shift` | 0.01 | SR regularisation: diagonal shift added to $\bar{S}$ |
| `cg_tol` | 1e-5 | CG solver convergence tolerance |

---

## Expected Output

Training logs print every 20 steps with one row per $\gamma$:

```
Step 0040 | γ=0.00  E/N=-0.4431(3)  σ²=0.0021  R̂=1.003  acc=0.52
Step 0040 | γ=0.10  E/N=-0.4299(4)  σ²=0.0034  R̂=1.005  acc=0.51
Step 0040 | γ=0.20  E/N=-0.4071(5)  σ²=0.0051  R̂=1.008  acc=0.49
...
```

Checkpoints are saved to `fnqs_j1j2_n20_ckpt.pkl` every 100 steps. Converged energies for the J1–J2 chain can be compared against exact diagonalisation benchmarks; at $\gamma = 0$ (pure Heisenberg) the ground-state energy per site is $E_0/N \approx -0.4431$ for open boundary conditions at N=20.
