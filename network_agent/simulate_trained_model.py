import numpy as np
import tensorflow as tf
import gymnasium as gym
from gymnasium.wrappers.flatten_observation import FlattenObservation
from dqn_agent import dqn_agent
import gym_environments
import pathlib
import sys

try:
    path = sys.argv[1]
    model_path = pathlib.Path(path)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'loss':dqn_agent.loss}
    )
except IndexError:
    print("Model path not provided")
except OSError:
    print("Invalid path provided")
else:
    # Creates the environment
    env = gym.make("MultiNodes-v0", render_mode="human", max_steps=50, min_range=1,
                    max_range=3, min_time=10, max_time=15)
    env = FlattenObservation(env)

    # Initializes the environment
    current_state, _ = env.reset()
    reward_sum = 0

    terminal_state = False
    while not terminal_state:
        # Uses the model to generate q values based on the current state
        q_values = model.predict(current_state[None, :])

        # Selects the optimal action based on the q values
        action = np.random.choice(
            np.where(q_values[0,:]==np.max(q_values[0,:]))[0]
        )
        current_state, reward, terminal_state, _, _ = env.step(action)
        reward_sum += reward
        print(f"Action {action}: Reward: {reward}")

    print(f"Reward sum: {reward_sum}")
    env.close()