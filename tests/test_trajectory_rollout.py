import jax
import jax.numpy as jnp
from flax import nnx

from modrax.env import DiscreteActionSpec, Env, EnvConfig, State, StateWithMetrics, StepOutput
from modrax.eval import eval_rollout
from modrax.network import Network
from modrax.rollout import trajectory_rollout, trajectory_to_transitions


class _Policy(Network):
    def policy(self, env_state, key):
        batch_size = env_state.obs.shape[0]
        return jnp.zeros(batch_size, dtype=jnp.int32), jnp.zeros(batch_size)


class _EvalPolicy(_Policy):
    def eval_policy(self, env_state, key):
        batch_size = env_state.obs.shape[0]
        return jnp.ones(batch_size, dtype=jnp.int32), jnp.zeros(batch_size)


class _StatefulPolicy(_Policy):
    def __init__(self):
        self.dropout = nnx.Dropout(0.5)
        self.state = nnx.Variable(jnp.ones(2))

    def reset(self):
        self.state[...] = 0


class _Env(Env):
    def __init__(self):
        self.obs_shape = (1,)
        self.action_spec = DiscreteActionSpec(5)
        super().__init__(EnvConfig())

    def _inner_reset_fn(self, key):
        return State(
            env_state=jnp.zeros((), dtype=jnp.int32),
            obs=jnp.zeros(1),
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool_),
            info={},
        )

    def _inner_step_fn(self, state, action, key):
        step = state.env_state + action + 1
        output = StepOutput(
            reward=jnp.ones(()),
            done=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
        )
        new_state = State(
            env_state=step,
            obs=jnp.full((1,), step, dtype=jnp.float32),
            action_mask=state.action_mask,
            info={},
        )
        return output, new_state


def _step(state, action, keys):
    batch_size = state.obs.shape[0]
    output = StepOutput(
        reward=jnp.zeros(batch_size),
        done=jnp.zeros(batch_size, dtype=jnp.bool_),
        truncation=jnp.zeros(batch_size, dtype=jnp.bool_),
        info={},
    )
    return output, state


def test_random_actions_use_per_step_keys():
    num_envs = 16
    num_steps = 8
    num_actions = 5
    key = jax.random.key(42)
    state = StateWithMetrics(
        env_state=jnp.zeros(num_envs),
        obs=jnp.zeros((num_envs, 1)),
        action_mask=jnp.ones((num_envs, num_actions), dtype=jnp.bool_),
        info={},
        episode_return=jnp.zeros(num_envs),
        episode_length=jnp.zeros(num_envs, dtype=jnp.int32),
    )

    _, trajectory = trajectory_rollout(
        _Policy(), _step, state, num_steps, key, random_action=True
    )

    step_keys = jax.random.split(key, num_steps)
    random_action_keys = jax.vmap(lambda step_key: jax.random.split(step_key, 3)[1])(step_keys)
    logits = jnp.zeros((num_envs, num_actions))
    expected_actions = jax.vmap(lambda action_key: jax.random.categorical(action_key, logits))(
        random_action_keys
    ).swapaxes(0, 1)

    assert jnp.array_equal(trajectory.actions, expected_actions)


def test_trajectory_to_transitions_adds_next_observations():
    num_envs = 3
    num_steps = 5
    env = _Env()
    state = env.reset(jax.random.split(jax.random.key(0), num_envs))

    final_state, trajectory = trajectory_rollout(_Policy(), env.step, state, num_steps, jax.random.key(1))
    transitions = trajectory_to_transitions(trajectory, final_state)

    assert transitions.obs.shape == (num_envs, num_steps, 1)
    assert transitions.next_obs.shape == (num_envs, num_steps, 1)
    assert transitions.actions.shape == (num_envs, num_steps)
    assert transitions.rewards.shape == (num_envs, num_steps)
    assert jnp.array_equal(transitions.next_obs[0, :, 0], jnp.arange(1, num_steps + 1))
    assert jnp.array_equal(transitions.truncations, trajectory.truncations)


def test_eval_rollout_is_batch_major():
    num_envs = 3
    num_steps = 5
    num_trajectories = 2

    _, trajectories = eval_rollout(
        _Env(),
        _EvalPolicy(),
        num_envs,
        jax.random.key(0),
        max_steps=num_steps,
        num_trajectories=num_trajectories,
    )

    assert trajectories.obs.shape == (num_trajectories, num_steps, 1)
    assert trajectories.env_state.shape == (num_trajectories, num_steps)
    assert jnp.array_equal(trajectories.env_state[0], 2 * jnp.arange(1, num_steps + 1))


def test_network_eval_sets_layer_mode_and_resets_state():
    network = _StatefulPolicy()

    network.eval()

    assert network.dropout.deterministic
    assert jnp.array_equal(network.state[...], jnp.zeros(2))
