import jax
import jax.numpy as jnp

from modrax.env import DiscreteActionSpec, Env, EnvConfig, State, StateWithMetrics, StepOutput
from modrax.network import Network
from modrax.rollout.eval_rollout import eval_rollout
from modrax.rollout.trajectory_rollout import trajectory_rollout
from modrax.rollout.transitions_rollout import transitions_rollout


class _Policy(Network):
    def policy(self, env_state, key):
        batch_size = env_state.obs.shape[0]
        return jnp.zeros(batch_size, dtype=jnp.int32), jnp.zeros(batch_size)


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
        step = state.env_state + 1
        output = StepOutput(
            reward=jnp.ones(()),
            done=jnp.bool_(False),
            truncation=jnp.bool_(False),
            info={},
        )
        new_state = State(
            env_state=step,
            obs=jnp.full((1,), step),
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


def test_transition_rollout_is_batch_major():
    num_envs = 3
    num_steps = 5
    state = StateWithMetrics(
        env_state=jnp.zeros(num_envs),
        obs=jnp.zeros((num_envs, 2)),
        action_mask=jnp.ones((num_envs, 4), dtype=jnp.bool_),
        info={},
        episode_return=jnp.zeros(num_envs),
        episode_length=jnp.zeros(num_envs, dtype=jnp.int32),
    )

    _, transitions = transitions_rollout(_Policy(), _step, state, num_steps, jax.random.key(0))

    assert transitions.obs.shape == (num_envs, num_steps, 2)
    assert transitions.actions.shape == (num_envs, num_steps)
    assert transitions.rewards.shape == (num_envs, num_steps)


def test_eval_rollout_is_batch_major():
    num_envs = 3
    num_steps = 5
    num_trajectories = 2

    _, trajectories = eval_rollout(
        _Env(),
        _Policy(),
        num_envs,
        jax.random.key(0),
        max_steps=num_steps,
        num_trajectories=num_trajectories,
    )

    assert trajectories.obs.shape == (num_trajectories, num_steps, 1)
    assert trajectories.env_state.shape == (num_trajectories, num_steps)
