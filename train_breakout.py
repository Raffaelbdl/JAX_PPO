import envpool
import wandb

from ppo.ppo_atari import PPO

from rl_tools.training import train_envpool_with_value

import numpy as np


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
    # wandb.init(entity="entity", project="ppo_breakout", config=config)
    train_envpool_with_value(config, envs, agent, use_wandb=False)


if __name__ == "__main__":
    main()
