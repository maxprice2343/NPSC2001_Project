import gymnasium as gym
from gymnasium import spaces
import numpy as np

WINDOW_SIZE = 512

class ProjEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, num_nodes=4):
        self.size = size
        self.num_nodes = num_nodes
        self.window_size = WINDOW_SIZE

        # Observations consist of agent location, node locations, and currently
        # active node
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "nodes": spaces.Sequence(spaces.Box(0, size - 1, shape=(2,), dtype=int)),
                "active": spaces.MultiBinary(num_nodes)
            }
        )
        # Agent can move up, down, left, or right
        self.action_space = spaces.Discrete(4)

        # Ensures the provided render mode is None (i.e., environment is not to
        # be rendered) or part of the array defining valid render types
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    # Initiates a new episode.
    def reset(self, seed=None):
        # Seeds the random number generator
        super().reset(seed=seed)

        # Randomly generates the agent location in the form [x,y]
        self._agent_location = self._generate_random_position()

        # Randomly generates node locations until they are not equal to the agent location
        # or other node locations
        for i in range(self.num_nodes):
            self._node_locations[i] = self._generate_random_position(
                excludes=(self._agent_location + self._node_locations[:i])
            )
        
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def _generate_random_position(self, excludes=[]):
        location = self.np_random.integers(0, self.size, size=2, dtype=int)

        while location in excludes:
            location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        return location

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "nodes": self._node_locations,
            "active": self._active_nodes
        }

    def _get_distance(self, node_number):
        return np.linalg.norm(self._agent_location - self._node_locations[node_number])

    def _action_to_direction(self, action):
        """
        Maps an action (integer from 0-3) to a direction (represented using a
        1D numpy array with 2 elements)
        """
        match action:
            case 0:
                direction = np.array([1,0])
            case 1:
                direction = np.array([0,1])
            case 2:
                direction = np.array([-1,0])
            case 3:
                direction = np.array([0,-1])

        return direction            
        