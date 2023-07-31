import numpy as np
import tensorflow as tf
import gymnasium as gym
from dqn_agent import dqn_agent
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
    env = gym.make("Network-v0", size=10, render_mode="human")

    # Initializes the environment
    current_state, _ = env.reset()

    terminal_state = False
    while not terminal_state:
        # Uses the model to generate q values based on the current state
        q_values = model.predict(current_state[None, :])

        # Selects the optimal action based on the q values
        action = np.random.choice(
            np.where(q_values[0,:]==np.max(q_values[0,:]))[0]
        )
        current_state, reward, terminal_state, _, _ = env.step(action)
    env.close()