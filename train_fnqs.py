"""
Generalized Stochastic Reconfiguration training loop for FNQS,
applied to the 1D J1-J2 Heisenberg chain with gamma = J2/J1.

FIXES vs the original version (see conversation for full rationale):
  - QGT is now nk.optimizer.qgt.QGTOnTheFly instead of QGTJacobianPyTree.
    QGTJacobianPyTree MATERIALIZES the (n_samples x n_params) Jacobian
    per state (for a non-holomorphic output like this model, as a
    real+imag pair). At N=20 (n_params~150k) with n_samples in the
    thousands, times R separate states all held in memory at once,
    this is what actually caused the OOM - not the physics. QGTOnTheFly
    never forms that matrix; it computes S @ v via composed JVP/VJP,
    so peak memory no longer scales with n_samples * n_params * R.
  - dtheta is globally norm-clipped before being applied. The failure
    in the original run was a single-step |dtheta| spike (one gamma's
    MC estimate went bad -> polluted the *shared* F_bar/S_bar across
    all R systems) that overflowed the log-amplitude head and produced
    permanent NaNs for the rest of the run. Clipping bounds the damage
    any one bad step can do to the shared parameters.
  - If forces, dtheta, or the CG solve ever produce a non-finite value
    despite clipping, that step's parameter update is skipped (old
    params kept, CG warm-start reset to zero) instead of silently
    corrupting theta. This is a per-step skip, not a run abort - the
    loop keeps going and will try again next step with fresh samples.
"""
import jax
jax.config.update("jax_enable_x64", True)  # SR/QGT solves want float64 throughout

import jax.numpy as jnp
import numpy as np
import netket as nk
from jax.flatten_util import ravel_pytree

from model import FNQS_J1J2


def build_chain_graph(N):
    edges = []
    for i in range(N):
        edges.append([i, (i + 1) % N, 0])  # J1 bonds (color 0)
        edges.append([i, (i + 2) % N, 1])  # J2 bonds (color 1)
    return nk.graph.Graph(edges=edges)


def make_apply_fixed_gamma(model, gamma):
    def f(params, sigma, **kwargs):
        g = jnp.full(sigma.shape[:-1] + (1,), gamma, dtype=sigma.dtype)
        x = jnp.concatenate([sigma, g], axis=-1)
        return model.apply(params, x, **kwargs)

    return f


def _global_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in leaves))


def _all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(all(np.all(np.isfinite(np.asarray(x))) for x in leaves))


class FNQSTrainer:
    def __init__(
        self,
        N,
        gammas,
        patch=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        n_samples=512,
        n_chains=32,
        n_discard_per_chain=16,
        diag_shift=1e-2,
        cg_tol=1e-6,
        cg_maxiter=200,
        chunk_size=None,
        max_dtheta_norm=5.0,   # NEW: global-norm clip applied every step
        seed=0,
    ):
        self.N = N
        self.gammas = list(gammas)
        self.R = len(self.gammas)
        self.diag_shift = diag_shift
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self.max_dtheta_norm = max_dtheta_norm

        self.graph = build_chain_graph(N)
        self.hi = nk.hilbert.Spin(s=0.5, N=N, total_sz=0)

        self.model = FNQS_J1J2(
            N=N, patch=patch, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, n_layers=n_layers,
        )

        key = jax.random.PRNGKey(seed)
        dummy_sigma = self.hi.random_state(key, 4)
        dummy_x = jnp.concatenate(
            [dummy_sigma, jnp.full((4, 1), self.gammas[0])], axis=-1
        )
        self.params = self.model.init(key, dummy_x)
        self.params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64), self.params
        )
        _, self.unravel = ravel_pytree(self.params)
        self.n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))

        self.states = []
        self.hamiltonians = []
        for i, g in enumerate(self.gammas):
            H = nk.operator.Heisenberg(hilbert=self.hi, graph=self.graph, J=[1.0, g])
            sampler = nk.sampler.MetropolisExchange(
                self.hi, graph=self.graph, n_chains=n_chains
            )
            vs = nk.vqs.MCState(
                sampler,
                apply_fun=make_apply_fixed_gamma(self.model, g),
                n_samples=n_samples,
                n_discard_per_chain=n_discard_per_chain,
                variables=self.params,
                seed=seed + i + 1,
                chunk_size=chunk_size,
            )
            self.states.append(vs)
            self.hamiltonians.append(H)

        self._x0 = None

    def _sync_params(self):
        for vs in self.states:
            vs.variables = self.params

    def step(self, lr=0.02, diag_shift=None):
        if diag_shift is None:
            diag_shift = self.diag_shift
        self._sync_params()

        F_list, S_ops, E_list, Evar_list = [], [], [], []
        Rhat_list, tau_list, err_list, accept_list = [], [], [], []
        for vs, H in zip(self.states, self.hamiltonians):
            vs.sample()
            E, forces = vs.expect_and_grad(H)
            F_list.append(jax.tree_util.tree_map(jnp.real, forces))
            # Matrix-free QGT: never materializes an (n_samples x n_params)
            # array, so memory no longer scales with n_samples * n_params * R.
            S_ops.append(
                nk.optimizer.qgt.QGTOnTheFly(vs, diag_shift=diag_shift)
            )
            E_list.append(E.mean.real)
            Evar_list.append(E.variance)
            Rhat_list.append(float(E.R_hat))
            tau_list.append(float(E.tau_corr))
            err_list.append(float(E.error_of_mean))
            accept_list.append(float(vs.sampler_state.acceptance))

        F_bar = jax.tree_util.tree_map(
            lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *F_list
        )

        def S_bar_matvec(v):
            applied = [S @ v for S in S_ops]
            return jax.tree_util.tree_map(
                lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *applied
            )

        if self._x0 is None:
            self._x0 = jax.tree_util.tree_map(jnp.zeros_like, F_bar)

        dtheta, info = jax.scipy.sparse.linalg.cg(
            S_bar_matvec, F_bar, x0=self._x0,
            tol=self.cg_tol, maxiter=self.cg_maxiter,
        )

        residual = jax.tree_util.tree_map(
            lambda a, b: a - b, S_bar_matvec(dtheta), F_bar
        )
        f_norm = float(_global_norm(F_bar))
        rel_residual = float(_global_norm(residual) / (f_norm + 1e-12))
        raw_dtheta_norm = float(_global_norm(dtheta))

        finite = (
            _all_finite(F_bar)
            and _all_finite(dtheta)
            and np.isfinite(raw_dtheta_norm)
            and np.isfinite(rel_residual)
        )

        if not finite:
            # Non-finite step: skip the parameter update entirely, reset
            # the CG warm start (a poisoned warm start would just corrupt
            # every future step's CG solve too), keep old params, and let
            # the caller know via 'skipped' so it can show/log it. This is
            # a per-step skip, not an abort - training continues.
            self._x0 = jax.tree_util.tree_map(jnp.zeros_like, F_bar)
            return {
                "energy": np.array(E_list),
                "energy_var": np.array(Evar_list),
                "R_hat": np.array(Rhat_list),
                "tau_corr": np.array(tau_list),
                "error_of_mean": np.array(err_list),
                "acceptance": np.array(accept_list),
                "cg_rel_residual": float("nan"),
                "dtheta_norm": 0.0,
                "skipped": True,
            }

        # Global-norm clip: bounds how much damage any single noisy/ill-
        # conditioned step can do to the *shared* parameters (all R
        # systems ride on the same theta, so one bad gamma can otherwise
        # blow up all of them at once).
        clip_scale = min(1.0, self.max_dtheta_norm / (raw_dtheta_norm + 1e-12))
        dtheta = jax.tree_util.tree_map(lambda d: d * clip_scale, dtheta)
        self._x0 = dtheta  # warm-start next step's CG with this (clipped) solution

        self.params = {
            "params": jax.tree_util.tree_map(
                lambda p, d: p - lr * d, self.params["params"], dtheta
            )
        }
        self._sync_params()

        return {
            "energy": np.array(E_list),
            "energy_var": np.array(Evar_list),
            "R_hat": np.array(Rhat_list),
            "tau_corr": np.array(tau_list),
            "error_of_mean": np.array(err_list),
            "acceptance": np.array(accept_list),
            "cg_rel_residual": rel_residual,
            "dtheta_norm": raw_dtheta_norm * clip_scale,
            "skipped": False,
        }

    def energy_per_gamma(self):
        self._sync_params()
        out = []
        for vs, H in zip(self.states, self.hamiltonians):
            vs.sample()
            out.append(vs.expect(H))
        return out
