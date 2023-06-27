import gymnasium as gym
from gymnasium import spaces
import numpy as np

WINDOW_SIZE = 512
NUM_NODES = 4

class ProjEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size
        self.window_size = WINDOW_SIZE

        # Observations consist of agent location, node locations, and currently
        # active node
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "nodes": spaces.Sequence(spaces.Box(0, size - 1, shape=(2,), dtype=int)),
                "active": spaces.MultiBinary(NUM_NODES)
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
    
    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "nodes": self._node_locations,
            "active": self._active_nodes
        }

    def _get_distance(self, node_number):
        return np.linalg.norm(self._agent_location - self._node_locations)
        return

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
        