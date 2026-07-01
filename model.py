"""
FNQS-style transformer ansatz for the 1D J1-J2 Heisenberg chain,
with gamma = J2/J1 as an O(1) coupling fed into the patch embedding.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, n_patches, d_model)
        d_head = self.d_model // self.n_heads
        qkv = nn.Dense(3 * self.d_model, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        b, n, _ = x.shape

        def split_heads(t):
            return t.reshape(b, n, self.n_heads, d_head).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) / jnp.sqrt(d_head)
        attn = nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.d_model)
        return nn.Dense(self.d_model, use_bias=False, name="proj")(out)


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        x = x + MultiHeadAttention(self.d_model, self.n_heads)(h)
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d_ff)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model)(h)
        return x + h


class FNQS_J1J2(nn.Module):
    """
    Input x has shape (..., N+1): the last column is gamma = J2/J1
    (broadcast to every sample), the first N columns are the spin
    configuration in {-1, +1}.

    Patch size `patch`: number of sites grouped into one token.
    gamma is concatenated (O(1) coupling case) to every patch before
    the linear embedding, exactly as in the FNQS paper's embedding
    strategy for scalar couplings.
    """
    N: int
    patch: int = 2
    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    n_layers: int = 2

    @nn.compact
    def __call__(self, x):
        sigma = x[..., : self.N]
        gamma = x[..., self.N :]                     # (..., 1)
        batch_shape = sigma.shape[:-1]

        n_patch = self.N // self.patch
        patches = sigma.reshape(batch_shape + (n_patch, self.patch))
        gamma_b = jnp.broadcast_to(
            gamma[..., None, :], batch_shape + (n_patch, gamma.shape[-1])
        )
        tokens = jnp.concatenate([patches, gamma_b], axis=-1)  # (..., n_patch, patch+1)

        # collapse any leading batch dims into one for the transformer, restore after
        lead = tokens.shape[:-2]
        tokens = tokens.reshape((-1,) + tokens.shape[-2:])

        h = nn.Dense(self.d_model, name="embed")(tokens)

        # simple learned relative-position bias via additive positional embedding
        pos = self.param(
            "pos_embed", nn.initializers.normal(0.02), (n_patch, self.d_model)
        )
        h = h + pos[None, :, :]

        for i in range(self.n_layers):
            h = EncoderBlock(self.d_model, self.n_heads, self.d_ff, name=f"block{i}")(h)

        z = jnp.sum(h, axis=1)  # sum pooling over patches -> (batch, d_model)

        # complex-valued output layer: log-amplitude (real) and phase (real),
        # combined into a complex log-psi. This is more numerically stable
        # than a literal complex Dense layer.
        log_amp = nn.Dense(1, name="log_amp_head")(z)[..., 0]
        phase = nn.Dense(1, name="phase_head")(z)[..., 0]

        out = log_amp + 1j * phase
        return out.reshape(lead)
