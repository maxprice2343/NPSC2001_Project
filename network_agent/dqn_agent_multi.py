import numpy as np
import tensorflow as tf
import gymnasium as gym
import gym_environments
from gymnasium.wrappers.flatten_observation import FlattenObservation
from replay_buffer import transition, replay_buffer

UPDATE_TARGET_NETWORK_PERIOD = 100

class dqn_agent_multi:
    def __init__(self, env, location_shape, time_shape, range_shape, state_dimension, action_dimension, gamma,
                 epsilon, num_episodes, replay_buffer_size, batch_size):
        self.env = FlattenObservation(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes

        self.location_shape = location_shape
        self.time_shape = time_shape
        self.range_shape = range_shape
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

        self.replay_buffer = replay_buffer(replay_buffer_size, batch_size)

        self.update_target_network_period = UPDATE_TARGET_NETWORK_PERIOD
        self.update_target_network_count = 0

        self.main_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.main_network.get_weights())

        self.sum_rewards_episode = []
        self.actions_append = []
    
    def create_network(self):
        """
        Creates the network layers and compiles it
        """
        # Location branch
        location_input = tf.keras.Input(shape=self.location_shape)
        x = tf.keras.layers.Dense(8, activation='relu')(location_input)
        x = tf.keras.layers.Dense(4, activation='relu')(x)
        self.location_model = tf.keras.Model(inputs=location_input, outputs=x)

        # Time branch
        time_input = tf.keras.Input(shape=self.time_shape)
        y = tf.keras.layers.Dense(8, activation='relu')(time_input)
        y = tf.keras.layers.Dense(4, activation='relu')(y)
        self.time_model = tf.keras.Model(inputs=time_input, outputs=y)

        # Range branch
        range_input = tf.keras.Input(shape=self.range_shape)
        z = tf.keras.layers.Dense(8, activation='relu')(range_input)
        z = tf.keras.layers.Dense(4, activation='relu')(z)
        self.range_model = tf.keras.Model(inputs=range_input, outputs=z)

        # Combines the outputs of the 3 layers into 1
        combined = tf.keras.layers.concatenate([x.output, y.output, z.output])

        # Creates the final 2 layers including the output layer
        final = tf.keras.layers.Dense(8, activation='relu')(combined)
        final = tf.keras.layers.Dense(self.action_dimension, activation='linear')(final)

        model = tf.keras.Model(inputs=[x.input, y.input, z.input], outputs=final)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=self.loss)

        return model
    
    def select_action(self, state, index):
        """
        Selects an action based on the current environment state
        """
        action = None

        # If it's the first epiisode, all actions are selected randomly
        if index < 1:
            action = np.random.randint(0, 4)
        else:
            # A random float in the range [0.0, 1.0)
            r = np.random.random()

            # After 200 episodes, start to slowly decrease epsilon, which
            # will decrease the number of actions selected randomly
            if index > 200:
                self.epsilon *= 0.999

            # If r is less than epsilon, then the action is selected randomly,
            # i.e., the agent will explore the environment
            # Otherwise, the agent will select the optimal action
            if r < self.epsilon:
                action = np.random.randint(0, 4)
            else:
                q_values = self.main_network.predict(state[None, :])
                # Returns the index where q_values has maximum values
                action = np.random.choice(
                    np.where(q_values[0,:]==np.max(q_values[0,:]))[0]
                )

        return action

    def train_network(self):
        """
        Trains the network
        """
        # If the replay buffer has at least 1 batch worth of elements then
        # we train the model
        batch_size = self.replay_buffer.batch_size
        if(self.replay_buffer.length() > batch_size):
            # Takes a random sample from the replay buffer
            sample_batch = self.replay_buffer.sample()

            current_state_batch = np.zeros(shape=(batch_size, self.state_dimension))
            next_state_batch = np.zeros(shape=(batch_size, self.state_dimension))

            # Enumerates through the batch of transitions and assigns
            for index, tran in enumerate(sample_batch):
                current_state_batch[index, :] = tran.state
                next_state_batch[index, :] = tran.next_state

            # Uses the target and main networks to predict q values
            q_next_state_target_network = self.target_network.predict(next_state_batch)
            q_current_state_main_network = self.main_network.predict(current_state_batch)

            input_network = current_state_batch
            output_network = np.zeros(shape=(batch_size, self.action_dimension))

            self.actions_append=[]
            for index, tran in enumerate(sample_batch):
                if tran.terminated:
                    y = tran.reward
                else:
                    y = tran.reward + self.gamma * np.max(q_next_state_target_network[index])
                
                self.actions_append.append(tran.action)

                # Stores the rewards from taking actions
                output_network[index, tran.action] = y

            # Trains the main network using the samples from the replay buffer
            self.main_network.fit(
                input_network,
                output_network,
                batch_size=batch_size,
                verbose = 0,
                epochs=100
            )

            # Periodically copies weights from the main network to the target network
            self.update_target_network_count += 1
            if(self.update_target_network_count > (self.update_target_network_period - 1)):
                self.target_network.set_weights(self.main_network.get_weights())
                print("Target network updated")
                self.update_target_network_count = 0
    
    def training_episodes(self):
        """
        Runs the simulation 'num_episodes' number of times and stores the
        results in the replay buffer for training the network.
        """
        for index in range(self.num_episodes):
            print(f"EPISODE: {index}\n")
            # Stores the rewards for each episode
            rewards_episodes = []

            (current_state, _) = self.env.reset()

            terminal_state = False
            while not terminal_state:
                # Select an action based on the current state
                action = self.select_action(current_state, index)

                # Steps the environment and stores the reward
                (next_state, reward, terminal_state, _, _) = self.env.step(action)
                print(f"next_state = {next_state}")
                # Creates a new transition object from the environment current
                # and next state
                new_transition = transition(current_state, action, reward, next_state, terminal_state)
                rewards_episodes.append(reward)

                # Adds the new transition to the replay buffer
                self.replay_buffer.append(new_transition)

                # Trains the network with the stored episodes
                self.train_network()

                # Sets the current state for the next step
                current_state = next_state
            
            self.sum_rewards_episode.append(np.sum(rewards_episodes))
    
    def loss(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Defines the loss function
        y_true is the target and y_pred is predicted by the network
        """
        s1, _ = y_true.shape

        # Indices is a 2d array that is used to index into y_true and y_pred
        indices = np.zeros(shape=(s1,2))
        for i in range(s1):
            indices[i, 0] = i
        indices[:,1] = self.actions_append

        # Calculates the mean squared error between y_true and y_pred
        loss = tf.keras.losses.mean_squared_error(
            tf.gather_nd(y_true, indices=indices.astype(int)),
            tf.gather_nd(y_pred, indices=indices.astype(int))
        )
        print(loss)
        return loss