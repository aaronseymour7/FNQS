"""
Minimal NetKet training script: FNQS (Vision-Transformer neural quantum state,
following Rende et al., Nat. Commun. 16, 7213 (2025)) for the 1D J1-J2
Heisenberg chain, N = 20 sites, trained with the *ensemble* Stochastic
Reconfiguration (SR) of Eqs. (15)-(18) of the paper so that a single set of
weights generalizes across a whole range of J2, not just one value.

Idea: the network wavefunction psi_theta(sigma | j2) takes the coupling j2
as an extra input. At every SR step we draw R couplings {j2_k} ~ P(j2),
build the corresponding Hamiltonians, sample each of them independently,
and combine the R gradients/QGTs into the ensemble-averaged G and S of
Eqs. (16)-(17) before doing one regularized SR update (Eq. 18). This lets
the trained model be evaluated (and extrapolated) at any j2 in supp(P),
including values never explicitly seen.

Changes vs the first version (for the R=4 -> R=9 regression):
  - DIAG_SHIFT is now annealed (starts higher, decays over training) instead
    of fixed. A fixed small shift became under-regularized once R changed the
    conditioning of the ensemble-averaged S, causing single-step blowups
    (visible in the old log as isolated positive-energy iterations).
  - The SR update is norm-clipped before being applied, so a single bad
    S_reg solve can no longer permanently corrupt `params` for the rest of
    training.
  - The per-10-iter printout now also reports mean e/site *per j2 tercile*
    (low third / mid third / high third of [J2_LOW, J2_HIGH]), using only
    the energies already computed this iteration -- no extra sampling, no
    mid-training tester calls. This is the diagnostic that would have shown
    the 0.4-0.6 region collapsing well before iteration 800.
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import netket as nk
import pickle

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------
# 1. Lattice, Hilbert space, and the family of Hamiltonians H(j2)
# --------------------------------------------------------------------------
L = 20
J1 = 1.0

nn_edges = [(i, (i + 1) % L, 0) for i in range(L)]   # nearest neighbours
nnn_edges = [(i, (i + 2) % L, 1) for i in range(L)]  # next-nearest neighbours
graph = nk.graph.Graph(edges=nn_edges + nnn_edges)

hi = nk.hilbert.Spin(s=0.5, N=L, total_sz=0.0)


def make_hamiltonian(j2, j1=J1):
    """Nearest + next-nearest neighbour Heisenberg chain for a given J2."""
    return nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[j1, j2])


# --------------------------------------------------------------------------
# 2. FNQS ansatz: a translation-invariant self-attention network that is
#    conditioned on the coupling j2 (1D, single-site patches, complex output)
# --------------------------------------------------------------------------
def log_cosh(x):
    sgn = -2 * jnp.signbit(x.real) + 1
    x = x * sgn
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class FMHA(nn.Module):
    """Translation-invariant factored multi-head self-attention (Eq. FMHA
    of the FNQS paper): the attention matrix is a learned circulant matrix,
    so it does not depend on the samples themselves."""
    d_model: int
    heads: int
    L: int

    @nn.compact
    def __call__(self, x):
        v = nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64)(x)
        v = v.reshape(x.shape[0], self.L, self.heads, -1).transpose(0, 2, 1, 3)

        J = self.param("J", nn.initializers.xavier_uniform(),
                        (self.heads, self.L), jnp.float64)
        # build a circulant (translation-invariant) attention matrix from J
        Jmat = jax.vmap(lambda j, s: jnp.roll(j, s), (None, 0), out_axes=1)(
            J, jnp.arange(self.L)
        )  # (heads, L, L)

        out = jnp.matmul(Jmat, v)                      # (batch, heads, L, d_eff)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], self.L, -1)
        return nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64)(out)


class EncoderBlock(nn.Module):
    d_model: int
    heads: int
    L: int

    @nn.compact
    def __call__(self, x):
        ln1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        ln2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        x = x + FMHA(self.d_model, self.heads, self.L)(ln1(x))
        ff = nn.Sequential([
            nn.Dense(4 * self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform()),
            nn.gelu,
            nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform()),
        ])
        x = x + ff(ln2(x))
        return x


class FNQS1D(nn.Module):
    """psi_theta(sigma | j2). `j2` is broadcast and concatenated to every
    site before the embedding, exactly as in Eq. (coups concatenation) of
    transformer_fnqs.py, so the same weights represent the whole family of
    ground states.

    j2 is read from a non-trainable "coupling" variable collection (instead
    of a plain positional argument) so that it can be swapped between MC
    samplings without any change to the static structure of the module
    (avoids jit recompilation every time we move to a different system)."""
    d_model: int = 16
    heads: int = 4
    num_layers: int = 2
    L: int = L

    @nn.compact
    def __call__(self, spins):
        j2 = self.variable("coupling", "j2", lambda: jnp.zeros((), jnp.float64)).value

        x = jnp.atleast_2d(spins)                                   # (batch, L)
        j2 = jnp.broadcast_to(jnp.asarray(j2, dtype=jnp.float64), (x.shape[0],))
        x = x[..., None]                                             # (batch, L, 1)
        j2_feat = jnp.broadcast_to(j2[:, None, None], (x.shape[0], self.L, 1))
        x = jnp.concatenate([x, j2_feat], axis=-1)                   # (batch, L, 2)

        x = nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform())(x)

        for _ in range(self.num_layers):
            x = EncoderBlock(self.d_model, self.heads, self.L)(x)

        z = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x.sum(axis=1))
        amp = nn.LayerNorm(use_scale=True, use_bias=True,
                            dtype=jnp.float64, param_dtype=jnp.float64)(
            nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform())(z))
        sign = nn.LayerNorm(use_scale=True, use_bias=True,
                             dtype=jnp.float64, param_dtype=jnp.float64)(
            nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform())(z))
        out = amp + 1j * sign
        return jnp.sum(log_cosh(out), axis=-1)


# --------------------------------------------------------------------------
# 3. Ensemble-SR training loop (Eqs. 15-18)
# --------------------------------------------------------------------------
R = 9                        # number of couplings sampled per SR step ("systems")
M_PER_SYSTEM = 512           # MC samples per system  ->  total M = R * M_PER_SYSTEM
N_ITERS = 1200
LR = 0.02
J2_LOW, J2_HIGH = 0.0, 0.6   # support of P(j2) the model is trained to cover

# -- j2 sampling distribution --
# NOTE: an earlier attempt biased 60% of each batch toward [0.3, 0.6] to
# give the hard/frustrated region more gradient signal. That caused
# persistent training instability (huge, erratic |delta| spikes recurring
# throughout the whole 800 iterations, energy oscillating and never
# breaking through toward convergence) -- almost certainly because most
# systems in each ensemble batch were now high-variance/hard-to-sample,
# with too few easy systems left to anchor stable G/S estimates at the
# current M_PER_SYSTEM. Reverted to plain uniform sampling, which is the
# configuration that produced the best, cleanly-converging result so far.
# If you want to revisit biasing later, do it much more mildly (e.g.
# HARD_REGION_FRAC ~0.15-0.2) and/or pair it with a higher M_PER_SYSTEM
# for the hard region specifically, rather than reintroducing it at 0.6.
HARD_REGION_LOW = 0.30
HARD_REGION_FRAC = 0.0


def sample_j2_batch(size):
    is_hard = np.random.rand(size) < HARD_REGION_FRAC
    hard_samples = np.random.uniform(HARD_REGION_LOW, J2_HIGH, size=size)
    full_samples = np.random.uniform(J2_LOW, J2_HIGH, size=size)
    return np.where(is_hard, hard_samples, full_samples)


# -- diag-shift annealing schedule --
# Starts high (strong regularization while the ensemble-averaged S is still
# poorly conditioned / far from any good basin) and decays toward the old
# fixed value. Tune DIAG_SHIFT_START/END/DECAY_ITERS if you still see
# blowups or if it feels over-regularized (energy plateauing too early).
DIAG_SHIFT_START = 1e-2
DIAG_SHIFT_END = 1e-3
DIAG_SHIFT_DECAY_ITERS = 400   # exponential decay reaches ~DIAG_SHIFT_END by this iter


def diag_shift_at(it):
    frac = min(it / DIAG_SHIFT_DECAY_ITERS, 1.0)
    # exponential interpolation in log-space between START and END
    log_shift = (1 - frac) * np.log(DIAG_SHIFT_START) + frac * np.log(DIAG_SHIFT_END)
    return float(np.exp(log_shift))


# -- SR update clipping --
# Guards against a single ill-conditioned S_reg solve corrupting params for
# the rest of training (this is what produced the isolated +0.39 / +1.22
# energy spikes at iters 250/330 in the original unclipped R=9 run).
#
# NOTE: 5.0 was an untuned guess and turned out to throttle almost every
# early-training step (raw |delta| routinely 20-80 for iters 0-500), which
# stalled convergence rather than just preventing blowups. Set this high
# enough that it only fires on genuine outliers, not routine large-but-
# legitimate SR steps. If you still see occasional isolated positive-energy
# iterations with this value, lower it; if [CLIPPED] fires on most
# iterations again, raise it further.
MAX_UPDATE_NORM = 100.0

model = FNQS1D()
rng = jax.random.PRNGKey(0)
dummy_spins = hi.random_state(jax.random.PRNGKey(1), 2)
variables = model.init(rng, dummy_spins)

sampler = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=M_PER_SYSTEM)

# one MCState per "system" (= one sampled j2), all sharing the same `params`;
# only the "coupling" collection differs from system to system.
vstates = [
    nk.vqs.MCState(sampler, model, variables=variables,
                    n_samples=M_PER_SYSTEM, n_discard_per_chain=8)
    for _ in range(R)
]

optimizer = optax.sgd(LR)
opt_state = optimizer.init(variables["params"])
import jax.flatten_util


def qgt_dense(vs):
    """Real quantum geometric tensor S(gamma_k) restricted to `params`,
    as a dense array (Eq. 17)."""
    return nk.optimizer.qgt.QGTJacobianDense(vs, diag_shift=0.0).to_dense()


def bin_by_tercile(j2_batch, energies, j2_low, j2_high):
    """Free diagnostic: split this iteration's already-computed energies
    into low/mid/high thirds of [j2_low, j2_high] and report per-bin means.
    No extra sampling -- reuses what expect_and_grad already computed."""
    edges = np.linspace(j2_low, j2_high, 4)  # 3 bins
    labels = ["low ", "mid ", "high"]
    out = []
    for b in range(3):
        mask = (j2_batch >= edges[b]) & (j2_batch <= edges[b + 1] + 1e-9)
        if mask.any():
            vals = np.asarray(energies)[mask]
            out.append(f"{labels[b]}[{edges[b]:.2f}-{edges[b+1]:.2f}]="
                       f"{vals.mean():+.4f}(n={mask.sum()})")
        else:
            out.append(f"{labels[b]}[{edges[b]:.2f}-{edges[b+1]:.2f}]=  n/a")
    return "  ".join(out)


if __name__ == "__main__":

    for it in range(N_ITERS):
        j2_batch = sample_j2_batch(R)
        diag_shift_it = diag_shift_at(it)

        G_sum = None
        S_sum = None
        energies = []
        unravel = None

        for k in range(R):
            j2_k = float(j2_batch[k])
            # condition system k on its own coupling; params stay shared
            vstates[k].variables = {
                **variables,
                "coupling": {"j2": jnp.asarray(j2_k, jnp.float64)},
            }

            H_k = make_hamiltonian(j2_k)

            e_k, G_k = vstates[k].expect_and_grad(H_k)
            S_k = qgt_dense(vstates[k])

            G_k_flat, unravel = jax.flatten_util.ravel_pytree(G_k)
            G_sum = G_k_flat if G_sum is None else G_sum + G_k_flat
            S_sum = S_k if S_sum is None else S_sum + S_k
            energies.append(e_k.mean.real / L)

        G = G_sum / R                      # ensemble-averaged gradient, Eq. (16)
        S = S_sum / R                      # ensemble-averaged QGT,      Eq. (17)
        S_reg = S + diag_shift_it * np.eye(S.shape[0])

        delta_flat = jnp.linalg.solve(S_reg, G)     # SR direction (Eq. 18, up to -eta)

        # -- clip the SR update norm before applying it --
        delta_norm = float(jnp.linalg.norm(delta_flat))
        if delta_norm > MAX_UPDATE_NORM:
            delta_flat = delta_flat * (MAX_UPDATE_NORM / delta_norm)

        delta = {"params": unravel(delta_flat)}

        updates, opt_state = optimizer.update(delta["params"], opt_state, variables["params"])
        new_params = optax.apply_updates(variables["params"], updates)
        variables = {**variables, "params": new_params}

        if it % 10 == 0:
            tercile_str = bin_by_tercile(j2_batch, energies, J2_LOW, J2_HIGH)
            clip_flag = " [CLIPPED]" if delta_norm > MAX_UPDATE_NORM else ""
            print(f"iter {it:4d}  diag_shift={diag_shift_it:.2e}  "
                  f"mean e/site={np.mean(energies):+.5f}  "
                  f"|delta|={delta_norm:.2f}{clip_flag}\n"
                  f"          {tercile_str}")

    with open("fnqs_variables.pkl", "wb") as f:
        pickle.dump(variables, f)
    print("Saved trained weights to fnqs_variables.pkl")
