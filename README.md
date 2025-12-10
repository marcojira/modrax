# Modrax
Modrax is a modular JAX reinforcement learning framework with composable components. Its goal is to simplify RL research with JAX environments. Built on the Flax NNX API, Modrax currently supports (more to come!):

- **Environments**: A unified API for interacting with JAX environments from the following suites:
  - [Craftax](https://github.com/MichaelTMatthews/Craftax)
  - [Gymnax](https://github.com/RobertTLange/gymnax)
  - [Octax](https://github.com/riiswa/octax)
  - [PGX](https://github.com/sotetsuk/pgx)
- **Networks**: Composable networks built from blocks. In particular, recurrent networks are supported. Blocks include:
  - Linear
  - MLP
  - RNN (and variants)
  - GatedTransformerXL (GTrXL)
- **Algorithms**: Various algorithms including:
  - PPO
  - DQN

## Installation
**Requires:** Python >= 3.10
```bash
git clone https://github.com/marcojira/modrax.git
pip install modrax/.

# To add octax (still in development):
# Follow instructions at https://github.com/riiswa/octax
```

## Quick Start
```python
"""Train PPO on MinAtar (250M steps, ~100s on an L40s)"""

from modrax.env import PGXEnvConfig
from modrax.network import MLP, BlockNetwork, BlockNetworkConfig, Linear, LinearConfig, MLPConfig
from modrax.types import OBS_SHAPE, NUM_ACTIONS
from modrax.optimizer import OptimizerConfig
from modrax.policy import softmax_policy
from modrax.rollout import RolloutConfig, rollout
from modrax.training import TrainConfig, train
from modrax.update import PPOConfig, ppo_update


def main():
    # Networks are built from blocks: BlockNetwork goes from encoders -> trunk -> heads
    # Use OBS_SHAPE/NUM_ACTIONS to use obs_shape/num_actions from environment
    network_config = BlockNetworkConfig(
        encoders={"obs": (MLP, MLPConfig(hidden_dims=(128,)), OBS_SHAPE)},
        encoder_dim=64,
        trunk=(Linear, LinearConfig()),
        trunk_dim=32,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(64, 64)), NUM_ACTIONS),
            "value": (MLP, MLPConfig(hidden_dims=(64, 64)), 1),
        },
    )

    train_config = TrainConfig(
        # Environment suite/environment
        env_config=PGXEnvConfig(env_name="minatar-asterix"),
        # Network class and config
        network_cls=BlockNetwork,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=3e-4, gradient_clip=0.5),
        # Specify rollout function/config
        rollout_fn=rollout,
        rollout_config=RolloutConfig(num_steps=128),
        # Specify network update function and config
        update_fn=ppo_update,
        update_config=PPOConfig(minibatch_size=128),
        # Specify policy function used for rollout
        policy_fn=softmax_policy,
        # Training parameters
        seed=0,
        num_envs=4096,
        total_steps=250_000_000,
        num_epochs=3,
        jit=True,
        # Save location
        save_path="out/examples/minatar-asterix",
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
```
## Project Structure
The quick start  example demonstrates the use of the different Modrax components and the built-in training scripts. Using the structure above, one can interchange
- the suite/environment by changing the env_config
- the network structure (e.g. to using a LSTM) by changing the network config and rollout function
- the training algorithm by changing the update function
and try out all the associated combinations.

However, Modrax was designed to be fairly flexible, allowing users to just use the components they need.

### Environments
Modrax provides a unifed wrapper/API for interacting with Jax environments, allowing for easier switching between suites with just a config change. All environments follow the same structure for initialization, reset, step, and action sampling.

```python
from modrax.envs import PGXEnvConfig

NUM_ENVS = 1024
key = jax.random.key(0)
reset_key, action_key, step_key = jax.random.split(key, 3)

# Create a PGX environment
env = Env(PGXEnvConfig(env_name="minatar-asterix", sticky_action_prob=0.005), jit=True)

# Reset
env_state = env.reset(jax.random.split(reset_key, NUM_ENVS))

# Step
actions = env.sample_action(action_key, NUM_ENVS)
step_keys = jax.random.split(step_key, NUM_ENVS)
step_output, env_state = env.step(env_state, actions, step_keys)
print(f"{step_output.reward}, {step_output.done}")
```

### Networks
Networks are built from composable blocks with consistent interfaces. Modrax supports two main network types:

- **BlockNetwork**: Standard feedforward network with the architecture `encoders → concatenate → trunk → heads`. Each input gets its own encoder block, and the outputs are concatenated and passed through the trunk and heads.
- **RecurrentNetwork**: Recurrent network for temporal dependencies with the architecture `encoders → recurrent block → heads`. Maintains recurrent state across rollouts.
  - Needs to be used with `recurrent_rollout` instead of `rollut`

Available blocks include:
- **Linear**: Simple linear transformation
- **MLP**: Multi-layer perceptron with configurable activation functions
- **NnxRNN**: Wrapper around Flax NNX RNN cells (LSTM, GRU, SimpleRNN)
- **GatedTransformerXL**: Transformer-based recurrent architecture with gating

```python
from flax import nnx
from modrax.network import RecurrentNetwork, RecurrentNetworkConfig, MLP, MLPConfig, NnxRNN, NnxRNNConfig
from modrax.types import OBS_SHAPE, NUM_ACTIONS

# Create a recurrent network with LSTM trunk
# Use OBS_SHAPE/NUM_ACTIONS to use obs_shape/num_actions
network_config = RecurrentNetworkConfig(
    encoders={"obs": (MLP, MLPConfig(hidden_dims=(128,)), OBS_SHAPE)},
    encoder_dim=64,
    recurrent=(NnxRNN, NnxRNNConfig(cell_type="lstm")),
    recurrent_dim=128,
    heads={
        "policy": (MLP, MLPConfig(hidden_dims=(64,)), NUM_ACTIONS),
        "value": (MLP, MLPConfig(hidden_dims=(64,)), 1),
    },
)

network = RecurrentNetwork(
    obs_shape=(4,),
    num_actions=4,
    config=network_config,
    rngs=nnx.Rngs(0),
)

# Forward pass
batch_size = 32
recurrent_state = network.init_recurrent_state(batch_size)
obs = jax.random.normal(jax.random.key(0), (batch_size, 4))
outputs, recurrent_state = network({"obs": obs}, recurrent_state)
# outputs = {"policy": [B, 4], "value": [B, 1]}
```

### Training
The `train` function takes in a `TrainingConfig` and
- Initializes environment/network/optimizer
- Runs the main training loop
  - Runs `rollout`
  - Update network using `RolloutData`
  - Logs metrics
  - Repeats

Many RL algorithms can be broken down into this loop of generating data through rollouts then updating the network. These algorithms can be implemented by providing the appropriate rollout and update functions.

#### Rollout
Rollout functions follow the following signature:
```python
class RolloutFn(Protocol):
    def __call__(
        self,
        network: Any,
        policy_fn: Callable,
        step_fn: Callable,
        env_state: Any,
        recurrent_state: Any,
        config: RolloutConfig,
        key: Key[Array, ""],
    ) -> tuple[Any, Any | None, RolloutData]: ...
```
For example:
```python
env_state, recurrent_state, data = rollout_fn(
    network,
    config.policy_fn,
    env.step,
    env_state,
    recurrent_state,
    config.rollout_config,
    rollout_key,
)
```

Currently Modrax supports the following 2 rollout functions:
- **`rollout`**: Standard rollout for feedforward networks
- **`recurrent_rollout`**: Rollout for recurrent networks, maintaining recurrent state across steps

#### Update
Once rollout data is collected, the data (`RolloutData`) is used by the update function to update the network parameters. Update functions follow the signature:
```python
class UpdateFn(Protocol):
    def __call__(
        self,
        network: Any,
        optimizer: Optimizer,
        data: RolloutData,
        config: Any,
        key: Key[Array, ""],
    ) -> tuple[Any, dict]: ...
```
For example:
```python
loss, infos = update_fn(
    network,
    optimizer,
    data,
    config.update_config,
    update_key,
)
```

- **`ppo_update`**: PPO with GAE advantage computation, policy/value clipping, and entropy regularization

## Examples
`examples/` contains example scripts demonstrating how to use Modrax for various RL tasks.
- `ppo_minatar.py`: Simples example of training a PPO agent on Minatar environments.
- `ppo_craftax_gtrxl`: A reimplementation of the SOTA GTrXL agent (from https://github.com/Reytuag/transformerXL_PPO_JAX) using Modrax components on Craftax.
