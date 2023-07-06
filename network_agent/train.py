import gymnasium as gym
from dqn_agent import DqnAgent
from replay_buffer import ReplayBuffer
import gym_examples

def collect_gameplay_experience(env, agent, buffer):
    """
    Initiates an episode of the environment. Actions are generated based on the
    agent's policy. The gameplay experience is stored in the buffer to be used
    for training.
    """
    # Initializes the environment and gets the initial environment
    state, _ = env.reset()
    done = False

    # Iterates until the environment is complete
    while not done:
        # Uses the agent's policy to produce an action given the current state
        action = agent.policy(state)
        # Applies the generated action to the environment
        next_state, reward, done, _, _= env.step(action)
        # Stores the gameplay experience (original state, resultant state,
        # reward, action taken, and done) in the buffer
        buffer.store_gameplay_experience(state, next_state, reward, action, done)
        state = next_state

def train_model(max_episodes=50000):
    """
    Trains a DQN agent to optimize the network time efficiency 
    """
    env = gym.make("Network-v0")
    agent = DqnAgent(4, 4)
    buffer = ReplayBuffer()

    for episode_count in range(max_episodes):
        # Starts 1 environment episode
        collect_gameplay_experience(env, agent, buffer)
        # Collects a random sample of gameplay experiences
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        # Updates the target network every 20 episodes
        if episode_count % 20 == 0:
            agent.update_target_network()
    
    env.close()

if __name__ == '__main__':
    train_model()
    