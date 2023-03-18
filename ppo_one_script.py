"""This is a all-in-one example of PPO implementation using JAX.

This follows https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ 
for Atari games and is evaluated on Breakout-v5 with Envpool.

This implementation should reach a score of 400 in less than 50M steps.

For more concise implementation, please check the `ppo` module.
"""

from collections import deque
from typing import Optional

import distrax as dx
from einops import rearrange
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import optax
import numpy as np
import rlax

import envpool
import gymnasium as gym

from rl_tools.update import update


# region Buffer
class SimpleOnPolicyBuffer:
    def __init__(self) -> None:
        self.keys = [
            "observations",
            "actions",
            "log_probs",
            "rewards",
            "dones",
            "next_observations",
            "values",
            "next_values",
        ]

    def reset(self):
        self.buffer = {key: [] for key in self.keys}

    def add(
        self,
        observation,
        action,
        log_prob,
        reward,
        done,
        next_observation,
        values,
        next_values,
    ):
        self.buffer["observations"].append(observation)
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["next_observations"].append(next_observation)
        self.buffer["values"].append(values)
        self.buffer["next_values"].append(next_values)

    def __len__(self):
        return len(self.buffer["rewards"])

    @staticmethod
    def get_batch_from_data(data: dict, idx):
        batch = {}
        for key in data.keys():
            batch[key] = data[key][idx]
        return batch


# endregion


# region Networks
class SharedEncoder(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = rearrange(observations, "a b c d -> a c d b") / 255.0

        x = jnn.relu(hk.Conv2D(32, 8, 4, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 4, 2, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 3, 1, w_init=self.w_init, b_init=self.b_init)(x))

        x = hk.Flatten()(x)

        x = jnn.relu(hk.Linear(512, w_init=self.w_init, b_init=self.b_init)(x))

        return x


class Actor(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.n = config["n_actions"]

        self.w_init = hk.initializers.Orthogonal(0.01)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        logits = hk.Linear(self.n, w_init=self.w_init, b_init=self.b_init)(embeds)

        return dx.Categorical(logits)


class Critic(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.w_init = hk.initializers.Orthogonal(1.0)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        return hk.Linear(1, w_init=self.w_init, b_init=self.b_init)(embeds)


def get_actor_critic(config: dict) -> tuple:
    """Returns a tuple

    (actor_fwd, actor_params, critic_fwd, critic_params)
    """
    key = jrd.PRNGKey(config["seed"])
    key1, key2 = jrd.split(key, 2)

    observations = jnp.zeros(config["observation_shape"])[None, ...]

    @hk.transform
    def actor_transformed(observations):
        embeds = SharedEncoder("shared_encoder")(observations)
        return Actor(config, "actor")(embeds)

    actor_fwd = actor_transformed.apply
    actor_params = actor_transformed.init(key1, observations)

    @hk.transform
    def critic_transformed(observations):
        embeds = SharedEncoder("shared_encoder")(observations)
        return Critic("critic")(embeds)

    critic_fwd = critic_transformed.apply
    critic_params = critic_transformed.init(key2, observations)

    return actor_fwd, actor_params, critic_fwd, critic_params


# endregion


# region PPO
class PPO:
    def __init__(self, config: dict, networks_factory=get_actor_critic):
        self.seed = config["seed"]
        self.key = jrd.PRNGKey(self.seed)

        (
            self.actor_fwd,
            self.actor_params,
            self.critic_fwd,
            self.critic_params,
        ) = networks_factory(config)

        self.params = hk.data_structures.merge(self.actor_params, self.critic_params)

        self.buffer_capacity = config["buffer_capacity"]

        self.gamma = config["gamma"]
        self.lambda_ = config["lambda"]
        self.epsilon = config["epsilon"]
        self.entropy_coef = config["entropy_coef"]
        self.critic_coef = config["critic_coef"]
        self.normalize = config["normalize"]

        self.loss_fn = get_ppo_loss(
            self.actor_fwd,
            self.critic_fwd,
            self.epsilon,
            self.entropy_coef,
            self.critic_coef,
            self.normalize,
        )
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

    def get_action(self, O):
        distributions = jax.jit(self.actor_fwd)(self.params, None, O)

        key = self._next_rng_key()
        actions, logps = distributions.sample_and_log_prob(seed=key)

        return np.array(actions, np.int32), logps

    def improve(self):
        self.init_metrics()

        data = self.prepare_data()
        idx = np.arange(len(data["rewards"]))

        for _ in range(self.n_samples_and_updates):
            idx = jrd.permutation(self._next_rng_key(), idx, independent=True)

            for i in range(self.n_minibatches):
                _idx = idx[i * self.batch_size : (i + 1) * self.batch_size]
                batch = SimpleOnPolicyBuffer.get_batch_from_data(data, _idx)

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
        key = self._next_rng_key()
        buffer = self.buffer.buffer

        all_observations = (
            buffer["observations"] + buffer["next_observations"][-1:]
        )  # T+1 * (N, size)
        all_observations = jnp.array(all_observations, jnp.float32)

        all_values = jax.vmap(lambda o: self.critic_fwd(self.params, key, o))(
            all_observations
        )[..., 0]

        rewards_t = jnp.array(buffer["rewards"], jnp.float32)  # T, N

        dones_t = jnp.array(buffer["dones"], jnp.bool_)  # T, N
        discounts_t = self.gamma * jnp.where(dones_t, 0.0, 1.0)

        def get_gae(all_values, rewards_t, discounts_t):
            advantages = rlax.truncated_generalized_advantage_estimation(
                rewards_t, discounts_t, self.lambda_, all_values, True
            )
            return advantages

        advantages = jax.vmap(get_gae, in_axes=(1, 1, 1), out_axes=1)(
            all_values, rewards_t, discounts_t
        )

        # returns = advantages + all_values[:-1]

        values_t = all_values[1:]

        def get_return(rewards_t, discounts_t, values_t):
            lambda_returns = rlax.lambda_returns(
                rewards_t, discounts_t, values_t, self.lambda_, True
            )
            return lambda_returns

        returns = jax.vmap(get_return, in_axes=(1, 1, 1), out_axes=1)(
            rewards_t, discounts_t, values_t
        )

        data = {}
        data["observations"] = rearrange(
            all_observations[:-1], "t n c h w -> (t n) c h w"
        )
        # data["observations"] = rearrange(all_observations[:-1], "t n s -> (t n) s")
        data["actions"] = rearrange(jnp.array(buffer["actions"]), "t n  -> (t n)")
        data["log_probs"] = rearrange(jnp.array(buffer["log_probs"]), "t n  -> (t n)")
        data["rewards"] = rearrange(rewards_t, "t n -> (t n)")
        data["discounts"] = rearrange(discounts_t, "t n -> (t n)")
        data["next_observations"] = rearrange(
            all_observations[1:], "t n c h w -> (t n) c h w"
        )
        # data["next_observations"] = rearrange(all_observations[1:], "t n s -> (t n) s")
        data["advantages"] = rearrange(advantages, "t n -> (t n)")
        data["returns"] = rearrange(returns, "t n -> (t n)")
        data["values"] = rearrange(all_values[:-1], "t n -> (t n)")

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
    actor_fwd,
    critic_fwd,
    epsilon: float,
    entropy_coef: float,
    critic_coef: float,
    normalize: bool,
):
    def actor_loss_fn(params, batch):
        O = batch["observations"]
        A = batch["actions"]
        Logp = batch["log_probs"]
        Adv = batch["advantages"]

        if normalize:
            Adv = (Adv - jnp.mean(Adv)) / (jnp.std(Adv) + 1e-8)

        new_Dist = actor_fwd(params, None, O)
        new_Logp = new_Dist.log_prob(A)  # B,
        Log_ratio = new_Logp - Logp
        Ratio = jnp.exp(Log_ratio)

        actor_loss = rlax.clipped_surrogate_pg_loss(Ratio, Adv, epsilon)
        entropy = jnp.mean(new_Dist.entropy())
        approx_kl = jax.lax.stop_gradient(jnp.mean((Ratio - 1) - Log_ratio))

        return actor_loss - entropy_coef * entropy, {
            "actor_loss": actor_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
        }

    def critic_loss_fn(params, batch):
        O = batch["observations"]
        Ret = batch["returns"]
        old_V = batch["values"]

        V = critic_fwd(params, None, O)[..., 0]
        V_clipped = old_V + jnp.clip(V - old_V, -epsilon, epsilon)

        critic_loss_unclipped = jnp.square(Ret - V)
        # critic_loss_clipped = jnp.square(Ret - V_clipped)
        # critic_loss = jnp.mean(jnp.fmax(critic_loss_unclipped, critic_loss_clipped))

        # return critic_coef * critic_loss, {"critic_loss": critic_loss}
        return critic_coef * critic_loss_unclipped, {
            "critic_loss": critic_loss_unclipped
        }

    def ppo_loss_fn(params, key, batch):
        actor_loss, actor_loss_dict = actor_loss_fn(params, batch)
        critic_loss, critic_loss_dict = critic_loss_fn(params, batch)

        total_loss = actor_loss + critic_loss
        loss_dict = actor_loss_dict
        loss_dict.update(critic_loss_dict)

        return jnp.mean(total_loss), loss_dict

    return ppo_loss_fn


# endregion


# region Train
def train_envpool(config: dict, envs: gym.Env, agent: PPO, use_wandb: bool = False):
    if not isinstance(envs, gym.Env):
        raise TypeError(
            "envs is not a gymnasium environment, please use env_type='gymnasium'"
        )

    if use_wandb:
        import wandb

    logs = {"steps": 0, "episodes": 0, "n_updates": 0, "episode_return": 0}

    n_envs = config["n_envs"]
    n_env_steps = config["n_env_steps"]

    observations, infos = envs.reset()
    episode_returns = np.zeros((n_envs,))
    for step in range(n_env_steps // n_envs):
        logs["steps"] = step * n_envs

        actions, log_probs = agent.get_action(observations)
        next_observations, rewards, dones, truncs, infos = envs.step(actions)

        episode_returns += rewards

        agent.buffer.add(
            observations, actions, log_probs, rewards, dones, next_observations
        )

        if agent.improve_condition:
            logs = logs | agent.improve()
            logs["n_updates"] += (
                config["n_samples_and_updates"] * config["n_minibatches"]
            )

        observations = next_observations

        for i, done in enumerate(dones):
            if done:
                logs["episode_return"] = episode_returns[i]
                episode_returns[i] = 0.0
                print(logs["steps"], logs["episode_return"])

                logs["episodes"] += 1

                if use_wandb:
                    wandb.log(logs)


# endregion


def main():
    config = {
        "name": "pongppo",
        "save_frequency": 20000,
        "seed": 0,
        "buffer_capacity": 128,
        "gamma": 0.99,
        "lambda": 0.95,
        "epsilon": 0.1,
        "entropy_coef": 0.01,
        "critic_coef": 0.5,
        "normalize": True,
        "learning_rate": 2.5e-4,
        "learning_rate_annealing": True,
        "n_env_steps": int(1e7),
        "n_envs": 8,
        "n_samples_and_updates": 4,
        "n_minibatches": 4,
        "max_gradient_norm": 0.5,
        "adam_epsilon": 1e-5,
    }

    envs = envpool.make(
        task_id="Breakout-v5",
        env_type="gymnasium",
        num_envs=config["n_envs"],
    )

    config["n_actions"] = envs.action_space.n
    config["observation_shape"] = envs.observation_space.shape

    agent = PPO(config)
    train_envpool(config, envs, agent, use_wandb=False)


if __name__ == "__main__":
    main()
