import numpy as np
import tensorflow as tf
import gymnasium as gym
from dqn_agent import dqn_agent
import pathlib

model_path = pathlib.Path("trained_model_temp.h5")

model = tf.keras.models.load_model(
    model_path,
    custom_objects={'loss':dqn_agent.loss}
)

env = gym.make("Network-v0", size=5, render_mode="rgb_array")

state, _ = env.reset()

video_length = 400
env = gym.wrappers.RecordVideo(env, 'stored_video', video_length=video_length)

for _ in range(10):
    terminal_state = False
    sum_obtained_rewards = 0
    while not terminal_state:
        q_values = model.predict(state[None, :])
        print(q_values)

        action = np.random.choice(
            np.where(q_values[0,:]==np.max(q_values[0,:]))[0]
        )
        current_state, reward, terminal_state, _, _ = env.step(action)
        sum_obtained_rewards += reward
    env.reset()

env.close()