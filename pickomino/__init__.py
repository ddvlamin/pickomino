from gymnasium.envs.registration import register

register(
     id="Pickomino-v0",
     entry_point="pickomino.env:PickominoEnv",
     max_episode_steps=300,
)