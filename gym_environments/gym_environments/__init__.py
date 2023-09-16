from gymnasium.envs.registration import register

register(
    id="Network-v0",
    entry_point="gym_environments.environments:NetworkEnv",
    max_episode_steps=300
)

register(
    id="MultiNodes-v0",
    entry_point="gym_environments.environments:MultiActiveEnv",
    max_episode_steps=300
)