"""
Generalized Stochastic Reconfiguration training loop for FNQS,
applied to the 1D J1-J2 Heisenberg chain with gamma = J2/J1.

This implements the paper's Algorithm: at each optimization step,
average the SR quantities (forces F and quantum geometric tensor S)
across R sampled couplings gamma, then take one natural-gradient
step on the SHARED network parameters theta.

QGT handling is matrix-free (QGTJacobianPyTree + CG), which is what
lets this scale to N=20 (n_params ~ 17k) without needing
n_samples > n_params for a well-conditioned *dense* solve, and without
ever forming an n_params x n_params matrix.
"""
import jax
jax.config.update("jax_enable_x64", True)  # SR/QGT solves want float64 throughout;
# also avoids float32/float64 dtype-mismatch inside jax.scipy.sparse.linalg.cg's
# while_loop, since netket promotes Jacobian/QGT internals to float64 anyway.

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
    """Wrap the FNQS model so netket sees a plain apply_fun(params, sigma)."""

    def f(params, sigma, **kwargs):
        g = jnp.full(sigma.shape[:-1] + (1,), gamma, dtype=sigma.dtype)
        x = jnp.concatenate([sigma, g], axis=-1)
        return model.apply(params, x, **kwargs)

    return f


class FNQSTrainer:
    def __init__(
        self,
        N,
        gammas,                 # fixed grid of R coupling values (J2/J1) to train on
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
        seed=0,
    ):
        self.N = N
        self.gammas = list(gammas)
        self.R = len(self.gammas)
        self.diag_shift = diag_shift
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter

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

        # one persistent MCState + Hamiltonian per gamma in the training grid
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
            )
            self.states.append(vs)
            self.hamiltonians.append(H)

        # warm-start guess for CG, carried across steps. Built lazily inside
        # step() from F_bar rather than here, because netket's forces/QGT
        # pytrees are float64 even when the model params are float32 (jax's
        # x64 promotion inside expect_and_grad) - jax.scipy.sparse.linalg.cg
        # requires x0's dtype to exactly match the matvec output's dtype.
        self._x0 = None

    def _sync_params(self):
        for vs in self.states:
            vs.variables = self.params

    def step(self, lr=0.02, diag_shift=None):
        """One generalized-SR step, averaged over all gamma in self.gammas.

        QGT is never materialized as a dense matrix: each S_r is a
        QGTJacobianPyTree LinearOperator (matrix-free matvec), the R of
        them are averaged into a single matvec closure, and the natural
        gradient direction is obtained with CG directly over the params
        pytree (no ravel/unravel needed for the solve itself).
        """
        if diag_shift is None:
            diag_shift = self.diag_shift
        self._sync_params()

        F_list, S_ops, E_list, Evar_list = [], [], [], []
        Rhat_list, tau_list, err_list, accept_list = [], [], [], []
        for vs, H in zip(self.states, self.hamiltonians):
            vs.sample()
            E, forces = vs.expect_and_grad(H)
            F_list.append(jax.tree_util.tree_map(jnp.real, forces))
            # diag_shift baked in per-gamma; since it's linear (shift*I),
            # averaging R copies of (S_r + shift*I) = S_bar + shift*I, exact.
            S_ops.append(
                nk.optimizer.qgt.QGTJacobianPyTree(vs, diag_shift=diag_shift)
            )
            E_list.append(E.mean.real)
            Evar_list.append(E.variance)
            Rhat_list.append(float(E.R_hat))
            tau_list.append(float(E.tau_corr))
            err_list.append(float(E.error_of_mean))
            accept_list.append(float(vs.sampler_state.acceptance))

        # F_bar: pytree average of the R force pytrees
        F_bar = jax.tree_util.tree_map(
            lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *F_list
        )

        # matrix-free averaged QGT action: v -> (1/R) sum_r S_r @ v
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
        self._x0 = dtheta  # warm-start next step's CG with this step's solution


        def _global_norm(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in leaves))

        residual = jax.tree_util.tree_map(
            lambda a, b: a - b, S_bar_matvec(dtheta), F_bar
        )
        rel_residual = float(_global_norm(residual) / (_global_norm(F_bar) + 1e-12))
        dtheta_norm = float(_global_norm(dtheta))

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
            "dtheta_norm": dtheta_norm,
        }

    def energy_per_gamma(self):
        self._sync_params()
        out = []
        for vs, H in zip(self.states, self.hamiltonians):
            vs.sample()
            out.append(vs.expect(H))
        return out
