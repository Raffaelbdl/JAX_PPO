from typing import Optional

import distrax
import einops
import haiku as hk
import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd


class SharedEncoder(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = einops.rearrange(observations, "a b c d -> a c d b") / 255.0

        x = jnn.relu(hk.Conv2D(32, 8, 4, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 4, 2, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 3, 1, w_init=self.w_init, b_init=self.b_init)(x))

        x = hk.Flatten()(x)

        x = jnn.relu(hk.Linear(512, w_init=self.w_init, b_init=self.b_init)(x))

        return x


class SharedEncoderLSTM(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = hk.initializers.Constant(0.0)

        self.lstm = hk.LSTM(128, "lstm")  # should add initialization
        self.hidden_states = None
        # w_init : hk.initializers.Orthogonal(1.0)
        # b_init : hk.initializers.Constant(0.0)

    def __call__(
        self,
        observations: jnp.ndarray,
        next_dones: jnp.ndarray,
        prev_hidden_states: jnp.ndarray = None,
    ) -> jnp.ndarray:
        x = einops.rearrange(observations, "a b c d -> a c d b") / 255.0

        x = jnn.relu(hk.Conv2D(32, 8, 4, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 4, 2, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 3, 1, w_init=self.w_init, b_init=self.b_init)(x))

        x = hk.Flatten()(x)

        x = jnn.relu(hk.Linear(512, w_init=self.w_init, b_init=self.b_init)(x))

        if prev_hidden_states is None:
            prev_hidden_states = self.lstm.initial_state(len(observations))

        # reinitialize state at episode end
        next_dones = next_dones[..., None]
        h = hk.LSTMState(
            hidden=prev_hidden_states[0] * (1.0 - next_dones),
            cell=prev_hidden_states[0] * (1.0 - next_dones),
        )
        x, h = self.lstm(x, h)

        return x, h


class Actor(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.n = config["n_actions"]

        self.w_init = hk.initializers.Orthogonal(0.01)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        logits = hk.Linear(self.n, w_init=self.w_init, b_init=self.b_init)(embeds)

        return distrax.Categorical(logits)


class Critic(hk.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

        self.w_init = hk.initializers.Orthogonal(1.0)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, embeds: jnp.ndarray) -> jnp.ndarray:
        return hk.Linear(1, w_init=self.w_init, b_init=self.b_init)(embeds)


class ContinousActor(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.n = config["n_actions"]

        self.encoder_w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.output_w_init = hk.initializers.Orthogonal(0.01)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations

        x = jnn.tanh(hk.Linear(64, w_init=self.encoder_w_init, b_init=self.b_init)(x))
        x = jnn.tanh(hk.Linear(64, w_init=self.encoder_w_init, b_init=self.b_init)(x))

        m_logits = hk.Linear(self.n, w_init=self.output_w_init, b_init=self.b_init)(x)

        std_logs = hk.get_parameter("std_logs", (1, self.n), init=jnp.zeros)
        std_logs = jnp.broadcast_to(std_logs, m_logits.shape)

        return distrax.Normal(m_logits, jnp.exp(std_logs))


class ContinuousCritic(hk.Module):
    def __init__(self, config: dict, name: Optional[str] = None):
        super().__init__(name)

        self.encoder_w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.output_w_init = hk.initializers.Orthogonal(1.0)
        self.b_init = hk.initializers.Constant(0.0)

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = observations

        x = jnn.tanh(hk.Linear(64, w_init=self.encoder_w_init, b_init=self.b_init)(x))
        x = jnn.tanh(hk.Linear(64, w_init=self.encoder_w_init, b_init=self.b_init)(x))

        return hk.Linear(1, w_init=self.output_w_init, b_init=self.b_init)(x)


def get_atari_actor_critic(config: dict) -> tuple:
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

    return (actor_fwd, critic_fwd), (actor_params, critic_params)


def get_continous_actor_critic(config: dict) -> tuple:
    """Returns a tuple

    (actor_fwd, actor_params, critic_fwd, critic_params)
    """
    key = jrd.PRNGKey(config["seed"])
    key1, key2 = jrd.split(key, 2)

    observations = jnp.zeros(config["observation_shape"])[None, ...]

    @hk.transform
    def actor_transformed(observations):
        return ContinousActor(config, "actor")(observations)

    actor_fwd = actor_transformed.apply
    actor_params = actor_transformed.init(key1, observations)

    @hk.transform
    def critic_transformed(observations):
        return ContinuousCritic(config, "critic")(observations)

    critic_fwd = critic_transformed.apply
    critic_params = critic_transformed.init(key2, observations)

    return actor_fwd, actor_params, critic_fwd, critic_params


def get_atari_actor_critic_lstm(config: dict) -> tuple:
    """Returns a tuple

    (actor_fwd, actor_params, critic_fwd, critic_params)
    """
    key = jrd.PRNGKey(config["seed"])
    key1, key2 = jrd.split(key, 2)

    observations = jnp.zeros(config["observation_shape"])[None, ...]
    dones = jnp.array([True])

    @hk.transform
    def actor_transformed(observations, dones, prev_lstm_states):
        embeds, lstm_states = SharedEncoderLSTM("shared_encoder")(
            observations, dones, prev_lstm_states
        )
        return Actor(config, "actor")(embeds), lstm_states

    actor_fwd = actor_transformed.apply
    actor_params = actor_transformed.init(key1, observations, dones, None)

    @hk.transform
    def critic_transformed(observations, dones, prev_lstm_states):
        embeds, lstm_states = SharedEncoderLSTM("shared_encoder")(
            observations, dones, prev_lstm_states
        )
        return Critic("critic")(embeds), lstm_states

    critic_fwd = critic_transformed.apply
    critic_params = critic_transformed.init(key2, observations, dones, None)

    @hk.transform
    def actor_critic_transformed(observations, dones, prev_lstm_states):
        embeds, lstm_states = SharedEncoderLSTM("shared_encoder")(
            observations, dones, prev_lstm_states
        )
        dists = Actor(config, "actor")(embeds)
        values = Critic("critic")(embeds)
        return dists, values, lstm_states

    actor_critic_fwd = actor_critic_transformed.apply

    return (actor_fwd, critic_fwd, actor_critic_fwd), (actor_params, critic_params)
