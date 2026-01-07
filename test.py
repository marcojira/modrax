import jax.numpy as jnp
from flax import nnx

B, T, D = 4, 10, 2

H = 4

x = jnp.arange(T)
x = jnp.expand_dims(x, axis=(0, 2))
x = jnp.tile(x, (B, 1, D))


y = jnp.zeros((B, T, H, D))


def make_window_array(x, window_size):
    H = window_size
    T = x.shape[1]

    indices = jnp.arange(T)[:, None] + jnp.arange(H)[None, :]
    y = x[:, indices, :]
    return y


y = make_window_array(x, H)
for i in range(T):
    print(y[0, i, :, 0])
# print(y)
