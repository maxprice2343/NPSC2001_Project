import tensorflow as tf
import numpy as np

class DqnAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_net = self._build_dqn_model()
        self.target_q_net = self._build_dqn_model()

    def policy(self, state: dict):
        """
        Takes an environment state and returns an action that the agent should
        take
        """
        # Concatenates the agent and active arrays into a 1 dimensional numpy
        # array
        con = np.concatenate((state['agent'], state['active']))
        # Converts the numpy array representing the environment state to a
        # tensor
        state_input = tf.convert_to_tensor(con, dtype=tf.float32)
        action_q = self.q_net(state_input)
        action = np.argmax(action_q.numpy()[0], axis=0)
        return action

    def train(self, batch):
        """
        Trains the DQN agent based on a batch of gameplay experiences
        """
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = batch
        current_q = self.q_net(state_batch)
        target_q = np.copy(current_q)
        next_q = self.target_q_net(next_state_batch)
        max_next_q = np.amax(next_q, axis=1)
        for i in range(state_batch.shape[0]):
            target_q[i][action_batch[i]] = reward_batch[i] if done_batch[i] else reward_batch[i] + 0.95 * max_next_q[i]
        result = self.q_net.fit(x=state_batch, y=target_q)

        return result.history['loss']

    def _build_dqn_model(self):
        q_net = tf.keras.Sequential()
        q_net.add(
            tf.keras.Input(shape=(self.state_dim))
        )
        q_net.add(
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        )
        q_net.add(
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')
        )
        q_net.add(
            tf.keras.layers.Dense(4, activation='linear', kernel_initializer='he_uniform')
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        q_net.compile(optimizer=optimizer, loss='mse')
        return q_net