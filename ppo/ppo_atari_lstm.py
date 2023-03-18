"""All-In-One PP0 implementation in JAX"""
from collections import deque

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrd
import optax
import numpy as np
import rlax

from rl_tools.agent import Agent
from rl_tools.buffer import SimpleOnPolicyBuffer
from rl_tools.update import update

from ppo.networks import get_atari_actor_critic_lstm


class PPO(Agent):
    def __init__(self, config: dict, networks_factory=get_atari_actor_critic_lstm):
        self.seed = config["seed"]
        self.key = jrd.PRNGKey(self.seed)

        fwd, params = networks_factory(config)
        self.actor_fwd, self.critic_fwd, self.actor_critic_fwd = fwd
        self.params = hk.data_structures.merge(*params)

        self.buffer_capacity = config["buffer_capacity"]

        self.gamma = config["gamma"]
        self.lambda_ = config["lambda"]

        self.loss_fn = get_ppo_loss(config, self.actor_critic_fwd)
        self.loss_fn = jax.jit(self.loss_fn)

        self.learning_rate = config["learning_rate"]
        self.learning_rate_annealing = config["learning_rate_annealing"]
        self.n_env_steps = config["n_env_steps"]
        self.n_envs = config["n_envs"]
        self.n_samples_and_updates = config["n_samples_and_updates"]
        self.n_minibatches = config["n_minibatches"]

        self.batch_size = self.buffer_capacity * self.n_envs // self.n_minibatches

        self.max_gradient_norm = config["max_gradient_norm"]
        self.adam_epsilon = config["adam_epsilon"]

        self.init_buffer()
        self.init_optimizer()

        self.n_envs = config["n_envs"]
        self.lstm_states = None

    def get_action(self, observations, dones):
        distributions, self.lstm_states = jax.jit(self.actor_fwd)(
            self.params, None, observations, dones, self.lstm_states
        )

        key = self._next_rng_key()
        actions, logps = distributions.sample_and_log_prob(seed=key)

        return np.array(actions, np.int32), logps

    def get_value_without_change_lstm_states(self, observations, dones, lstm_states):
        values, _ = jax.jit(self.critic_fwd)(
            self.params, None, observations, dones, lstm_states
        )
        return values

    def reset_lstm_states(self):
        self.lstm_states = None

    def improve(self):
        self.init_metrics()

        data = self.prepare_data()
        idx = np.arange(self.n_envs)
        n_envs_per_batch = self.n_envs // self.n_minibatches
        for e in range(self.n_samples_and_updates):
            idx = jrd.permutation(self._next_rng_key(), idx, independent=True)

            for i in range(self.n_minibatches):
                _idx = idx[i * n_envs_per_batch : (i + 1) * n_envs_per_batch]
                batch = SimpleOnPolicyBuffer.get_time_batch_from_data(data, _idx)

                self.params, self.opt_state, (total_loss, loss_dict) = update(
                    self.params,
                    self._next_rng_key(),
                    batch,
                    self.loss_fn,
                    self.optimizer,
                    self.opt_state,
                )

                self.metrics["total_loss"].append(np.array(total_loss))
                self.metrics["actor_loss"].append(np.array(loss_dict["actor_loss"]))
                self.metrics["critic_loss"].append(np.array(loss_dict["critic_loss"]))
                self.metrics["entropy"].append(np.array(loss_dict["entropy"]))
                self.metrics["approx_kl"].append(np.array(loss_dict["approx_kl"]))

        logs = {
            "total_loss": np.mean(self.metrics["total_loss"]),
            "actor_loss": np.mean(self.metrics["actor_loss"]),
            "critic_loss": np.mean(self.metrics["critic_loss"]),
            "entropy": np.mean(self.metrics["entropy"]),
            "approx_kl": np.mean(self.metrics["approx_kl"]),
        }

        self.buffer.reset()

        return logs

    @property
    def improve_condition(self):
        return len(self.buffer) >= self.buffer_capacity

    def prepare_data(self) -> dict:
        buffer = self.buffer.buffer

        # Compute advantages
        all_values = jnp.array(
            buffer["values"] + buffer["next_values"][-1:], jnp.float32
        )[..., 0]
        rewards_t = jnp.array(buffer["rewards"], jnp.float32)  # T, N
        dones_t = jnp.array(buffer["dones"], jnp.bool_)  # T, N
        discounts_t = self.gamma * jnp.where(dones_t, 0.0, 1.0)

        def get_gae(all_values, rewards_t, discounts_t):
            advantages = rlax.truncated_generalized_advantage_estimation(
                rewards_t, discounts_t, self.lambda_, all_values, True
            )
            return advantages

        advantages = jax.vmap(jax.jit(get_gae), in_axes=(1, 1, 1), out_axes=1)(
            all_values, rewards_t, discounts_t
        )

        returns = advantages + all_values[:-1]

        data = {}
        data["observations"] = jnp.array(buffer["observations"])
        data["actions"] = jnp.array(buffer["actions"])
        data["log_probs"] = jnp.array(buffer["log_probs"])
        data["rewards"] = rewards_t
        data["discounts"] = discounts_t
        data["next_observations"] = jnp.array(buffer["next_observations"])
        data["advantages"] = advantages
        data["returns"] = returns
        data["values"] = all_values[:-1]
        data["dones"] = dones_t

        return data

    def init_buffer(self) -> None:
        self.buffer = SimpleOnPolicyBuffer()
        self.buffer.reset()

    def init_optimizer(self) -> None:
        lr = self.learning_rate

        if self.learning_rate_annealing:
            n_updates = (
                self.n_env_steps
                // self.buffer_capacity
                // self.n_envs
                * self.n_samples_and_updates
                * self.n_minibatches
            )
            lr = optax.linear_schedule(lr, 0.0, n_updates, 0)

        @optax.inject_hyperparams
        def optimizer(learning_rate, eps, max_grad_norm):
            return optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate, eps=eps),
            )

        self.optimizer = optimizer(lr, self.adam_epsilon, self.max_gradient_norm)
        self.opt_state = self.optimizer.init(self.params)

    def init_metrics(self) -> None:
        self.metrics = {
            "total_loss": deque(),
            "actor_loss": deque(),
            "critic_loss": deque(),
            "entropy": deque(),
            "approx_kl": deque(),
        }

    def _next_rng_key(self):
        self.key, key1 = jrd.split(self.key)
        return key1


def get_ppo_loss(
    config: dict,
    actor_critic_fwd,
):
    def actor_loss_fn(batch, new_log_probs, entropy):
        log_probs = batch["log_probs"]
        advantages = batch["advantages"]

        if config["normalize"]:
            advantages = advantages - jnp.mean(advantages)
            advantages /= jnp.std(advantages) + 1e-8

        log_ratios = (new_log_probs - log_probs).ravel()
        ratios = jnp.exp(log_ratios)

        advantages = advantages.ravel()

        actor_loss = rlax.clipped_surrogate_pg_loss(
            ratios, advantages, config["epsilon"]
        )
        entropy = jnp.mean(entropy)
        approx_kl = jax.lax.stop_gradient(jnp.mean((ratios - 1) - log_ratios))

        return actor_loss - config["entropy_coef"] * entropy, {
            "actor_loss": actor_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
        }

    def critic_loss_fn(batch, new_values):
        returns = batch["returns"]
        values = batch["values"]

        clipped_values = values + jnp.clip(
            new_values - values, -config["epsilon"], config["epsilon"]
        )

        critic_loss_unclipped = jnp.square(returns - new_values)
        critic_loss_clipped = jnp.square(returns - clipped_values)
        critic_loss = jnp.mean(jnp.fmax(critic_loss_unclipped, critic_loss_clipped))

        return config["critic_coef"] * critic_loss, {"critic_loss": critic_loss}

    def ppo_loss_fn(params, key, batch):
        observations = batch["observations"]
        dones = batch["dones"]
        actions = batch["actions"]

        # centralize computation
        lstm_states = None
        new_log_probs = []
        entropy = []
        new_values = []
        for t in range(len(observations)):
            dists, values, lstm_states = actor_critic_fwd(
                params, None, observations[t], dones[t], lstm_states
            )
            new_log_probs.append(dists.log_prob(actions[t]))
            entropy.append(dists.entropy())
            new_values.append(values)

        new_log_probs = jnp.array(new_log_probs)
        entropy = jnp.array(entropy)
        new_values = jnp.array(new_values)[..., 0]

        actor_loss, actor_loss_dict = actor_loss_fn(batch, new_log_probs, entropy)
        critic_loss, critic_loss_dict = critic_loss_fn(batch, new_values)

        total_loss = actor_loss + critic_loss
        loss_dict = actor_loss_dict
        loss_dict.update(critic_loss_dict)

        return jnp.mean(total_loss), loss_dict

    return ppo_loss_fn
