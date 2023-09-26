import numpy as np
import gymnasium as gym
import gym_environments

def determine_optimal_move(state):
    agent_location = state["agent"]
    node_information = state["node_information"]
    min_time = np.min(node_information[:, 3])
    least_time = 100
    idx = 0
    for i, node_info in enumerate(node_information):
        if node_info[3] < least_time and node_info[3] != -1:
            least_time = node_info[3]
            idx = i
    node_to_seek_location = np.array([node_information[idx][0], node_information[idx][1]])
    x = node_to_seek_location - agent_location
    direction = np.random.randint(0, 2)
    if direction == 0 and x[direction] > 0:
        movement = 0
    elif direction == 0 and x[direction] < 0:
        movement = 2
    elif direction == 1 and x[direction] > 0:
        movement = 1
    else:
        movement = 3
    return movement

if __name__ == '__main__':
    env = gym.make("MultiNodes-v0", render_mode="human", max_steps=50, min_range=1,
               max_range=3, min_time=15, max_time=20)
    
    current_state, _ = env.reset()

    terminal_state = False
    reward_sum = 0
    while not terminal_state:
        movement = determine_optimal_move(current_state)

        current_state, reward, terminal_state, _, _ = env.step(movement)
        reward_sum += reward
    env.close()
    print(f"Reward Sum: {reward_sum}")