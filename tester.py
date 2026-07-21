"""
Tester: evaluate the ensemble-SR-trained FNQS model (see fnqs_train.py) against
exact diagonalization (ED) across a grid of J2 values, both inside the training
support [J2_LOW, J2_HIGH] (interpolation) and outside it (extrapolation).

Usage
-----
1. Run this in the *same process* right after training, so `variables` and
   `model` are already in scope, OR
2. Save `variables` to disk after training (see `save_variables` /
   `load_variables` below) and run this script standalone, importing the
   model class from your training script.

This script:
  - builds H(j2) for a grid of test couplings
  - runs exact diagonalization (Lanczos) for the exact ground energy/site
  - samples the trained NQS at each j2 and estimates energy/site + error bar
  - reports absolute/relative error and the NQS relative variance
    (a proxy for how well-converged/well-generalized the ansatz is there)
  - saves a CSV of results and a plot of NQS vs exact energy vs j2
"""

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import pandas as pd

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------
# Model / Hamiltonian definitions + trained weights.
# --------------------------------------------------------------------------
# We import the *definitions* only (model class, Hilbert space, graph,
# make_hamiltonian, J2_LOW/J2_HIGH) from the training module. As long as the
# training loop in fnqs_train.py is wrapped in `if __name__ == "__main__":`,
# this import is safe -- it will NOT re-run training.
from paper_train import model, hi, graph, L, J1, J2_LOW, J2_HIGH, make_hamiltonian

# Load the already-trained weights saved by fnqs_train.py, e.g. via
#   pickle.dump(variables, open("fnqs_variables.pkl", "wb"))
# at the end of the training script (see accompanying note).
import pickle

VARIABLES_PATH = "fnqs_variables.pkl"
with open(VARIABLES_PATH, "rb") as f:
    variables = pickle.load(f)
print(f"Loaded trained weights from {VARIABLES_PATH}")


# --------------------------------------------------------------------------
# 1. Evaluation helper: sample the NQS at a fixed j2 and measure <H(j2)>
# --------------------------------------------------------------------------
def evaluate_nqs_at_j2(j2_test, variables, n_samples=4096, n_discard=16, seed=0):
    """Sample psi_theta( . | j2_test) and measure energy/site for H(j2_test)."""
    sampler_eval = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=n_samples)
    vs = nk.vqs.MCState(
        sampler_eval,
        model,
        variables={**variables, "coupling": {"j2": jnp.asarray(j2_test, jnp.float64)}},
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
        seed=seed,
    )
    H_test = make_hamiltonian(j2_test)
    stats = vs.expect(H_test)

    e_mean = float(stats.mean.real) / L
    e_err = float(stats.error_of_mean) / L
    # relative variance of the local energy estimator, a rough proxy for how
    # well-converged / well-generalized the ansatz is at this j2
    rel_var = float(stats.variance) / (L * abs(e_mean) + 1e-12) ** 2

    return e_mean, e_err, rel_var


# --------------------------------------------------------------------------
# 2. Exact diagonalization helper
# --------------------------------------------------------------------------
def exact_energy_per_site(j2_test):
    """Ground-state energy per site of H(j2_test) via Lanczos ED.

    Feasible for N=20 with total_sz=0 (Hilbert dim ~ C(20,10) ~ 1.8e5)."""
    H_test = make_hamiltonian(j2_test)
    eigvals = nk.exact.lanczos_ed(H_test, k=1, compute_eigenvectors=False)
    return float(eigvals[0]) / L


# --------------------------------------------------------------------------
# 3. Sweep: interpolation grid inside [J2_LOW, J2_HIGH] + extrapolation points
# --------------------------------------------------------------------------
def build_test_grid(j2_low, j2_high, n_interp=9, n_extrap_each_side=2, margin=0.15):
    interp = np.linspace(j2_low, j2_high, n_interp)
    extrap_below = np.linspace(j2_low - margin, j2_low - 1e-3, n_extrap_each_side)
    extrap_above = np.linspace(j2_high + 1e-3, j2_high + margin, n_extrap_each_side)
    grid = np.concatenate([extrap_below, interp, extrap_above])
    return np.round(grid, 4)


def run_comparison(j2_grid, variables, n_samples=4096, n_discard=16, verbose=True):
    rows = []
    sampler_eval = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=n_samples)
    vs = nk.vqs.MCState(
        sampler_eval, model,
        variables={**variables, "coupling": {"j2": jnp.asarray(j2_grid[0], jnp.float64)}},
        n_samples=n_samples,
        n_discard_per_chain=1000,   # long cold-start burn-in, once
    )

    for j2 in j2_grid:
        e_exact = exact_energy_per_site(j2)

        # warm-start: reuse the *previous* j2's chain state, just swap coupling
        vs.variables = {**variables, "coupling": {"j2": jnp.asarray(j2, jnp.float64)}}
        vs.sample(n_discard_per_chain=200)   # short re-equilibration between nearby j2's

        stats = vs.expect(make_hamiltonian(j2))
        e_nqs = float(stats.mean.real) / L
        e_err = float(stats.error_of_mean) / L
        rel_var = float(stats.variance) / (L * abs(e_nqs) + 1e-12) ** 2

        abs_err = abs(e_nqs - e_exact)
        rel_err = abs_err / abs(e_exact)
        in_support = J2_LOW <= j2 <= J2_HIGH

        rows.append(
            dict(
                j2=j2,
                in_support=in_support,
                e_exact=e_exact,
                e_nqs=e_nqs,
                e_nqs_err=e_err,
                abs_err=abs_err,
                rel_err=rel_err,
                nqs_rel_variance=rel_var,
            )
        )

        if verbose:
            tag = "interp" if in_support else "EXTRAP"
            print(
                f"j2={j2:+.4f} [{tag}]  "
                f"exact={e_exact:.6f}  nqs={e_nqs:.6f}+-{e_err:.6f}  "
                f"abs_err={abs_err:.2e}  rel_err={rel_err:.2e}  "
                f"rel_var={rel_var:.2e}"
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Plot NQS vs exact energy across j2
# --------------------------------------------------------------------------
def plot_comparison(df, outpath="fnqs_vs_exact.png"):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.axvspan(J2_LOW, J2_HIGH, color="tab:blue", alpha=0.08, label="training support")
    ax1.plot(df["j2"], df["e_exact"], "k-", lw=1.5, label="Exact (Lanczos)")
    ax1.errorbar(
        df["j2"], df["e_nqs"], yerr=df["e_nqs_err"],
        fmt="o", color="tab:red", ms=5, capsize=3, label="FNQS (ensemble-SR)",
    )
    ax1.set_ylabel("Energy / site")
    ax1.legend()
    ax1.set_title(f"FNQS vs exact diagonalization, N={L} J1-J2 Heisenberg chain")

    ax2.axvspan(J2_LOW, J2_HIGH, color="tab:blue", alpha=0.08)
    ax2.semilogy(df["j2"], df["rel_err"], "o-", color="tab:green", ms=5)
    ax2.set_xlabel("J2 / J1")
    ax2.set_ylabel("Relative error\n|E_nqs - E_exact| / |E_exact|")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    N_SAMPLES = 4096       # increase for tighter NQS error bars
    N_DISCARD = 16

    j2_grid = build_test_grid(J2_LOW, J2_HIGH, n_interp=9, n_extrap_each_side=2)

    # Always include the well-known frustration/dimerization point of the
    # N->inf J1-J2 chain (~0.2411) if it's not already in the grid.
    landmark = 0.2411
    if not np.any(np.isclose(j2_grid, landmark, atol=1e-2)):
        j2_grid = np.sort(np.append(j2_grid, landmark))

    df = run_comparison(j2_grid, variables, n_samples=N_SAMPLES, n_discard=N_DISCARD)

    df.to_csv("fnqs_vs_exact_results.csv", index=False)
    print("\nSaved results to fnqs_vs_exact_results.csv")
    print(df.to_string(index=False))

    print(f"\nMean relative error (in-support):     "
          f"{df.loc[df.in_support, 'rel_err'].mean():.3e}")
    if (~df.in_support).any():
        print(f"Mean relative error (extrapolation):  "
              f"{df.loc[~df.in_support, 'rel_err'].mean():.3e}")

    try:
        plot_comparison(df)
    except ImportError:
        print("matplotlib not installed -- skipping plot, CSV results still saved.")
