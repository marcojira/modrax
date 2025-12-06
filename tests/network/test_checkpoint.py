import os
import tempfile

import jax
import jax.numpy as jnp
from flax import nnx

from modrax.network.base import Network


class SimpleMLP(Network):
    """Simple MLP for testing checkpoint functionality."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.layer2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x):
        x = self.layer1(x)
        x = jax.nn.relu(x)
        x = self.layer2(x)
        return x


def test_network_save():
    """Test that Network.save() creates checkpoint directory with expected structure."""
    # Create a simple MLP
    rngs = nnx.Rngs(0)
    mlp = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5, rngs=rngs)

    # Test forward pass works
    x = jnp.ones((2, 10))
    output = mlp(x)
    assert output.shape == (2, 5)

    # Save checkpoint to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint")
        mlp.save(checkpoint_path)

        # Verify checkpoint directory structure
        assert os.path.exists(checkpoint_path)
        assert os.path.exists(os.path.join(checkpoint_path, "params"))

        # Load checkpoint and test output
        loaded_mlp = mlp.load_checkpoint(checkpoint_path)
        loaded_output = loaded_mlp(x)

        assert jnp.allclose(output, loaded_output)
