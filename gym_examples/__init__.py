from gymnasium.envs.registration import register

register(
    id="Network-v0",
    entry_point="gym_examples.environment:NetworkEnv",
    max_episode_steps=300
)