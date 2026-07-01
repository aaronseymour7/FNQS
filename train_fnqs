"""
Generalized Stochastic Reconfiguration training loop for FNQS,
applied to the 1D J1-J2 Heisenberg chain with gamma = J2/J1.

This implements the paper's Algorithm: at each optimization step,
average the SR quantities (forces F and quantum geometric tensor S)
across R sampled couplings gamma, then take one natural-gradient
step on the SHARED network parameters theta.
"""
import jax
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
        seed=0,
    ):
        self.N = N
        self.gammas = list(gammas)
        self.R = len(self.gammas)
        self.diag_shift = diag_shift

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
        _, self.unravel = ravel_pytree(self.params)

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

    def _sync_params(self):
        for vs in self.states:
            vs.variables = self.params

    def step(self, lr=0.02):
        self._sync_params()

        F_list, S_list, E_list = [], [], []
        for vs, H in zip(self.states, self.hamiltonians):
            vs.sample()
            E, forces = vs.expect_and_grad(H)
            F_flat, _ = ravel_pytree(forces)
            S = vs.quantum_geometric_tensor(
                nk.optimizer.qgt.QGTJacobianDense
            ).to_dense()
            F_list.append(jnp.real(F_flat))
            S_list.append(jnp.real(S))
            E_list.append(E.mean.real)

        F_bar = jnp.mean(jnp.stack(F_list), axis=0)
        S_bar = jnp.mean(jnp.stack(S_list), axis=0)

        # regularized natural-gradient solve via truncated pseudo-inverse
        # (more robust than a plain solve when S is estimated from a finite
        # sample and is close to singular / ill-conditioned)
        S_reg = S_bar + self.diag_shift * jnp.eye(S_bar.shape[0])
        dtheta, *_ = jnp.linalg.lstsq(S_reg, F_bar, rcond=1e-6)

        theta_flat, _ = ravel_pytree(self.params)
        theta_flat = theta_flat - lr * dtheta
        self.params = self.unravel(theta_flat)
        self._sync_params()

        return np.array(E_list)

    def energy_per_gamma(self):
        self._sync_params()
        out = []
        for vs, H in zip(self.states, self.hamiltonians):
            vs.sample()
            out.append(vs.expect(H))
        return out
