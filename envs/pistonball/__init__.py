from gym.envs.registration import register
register(
    id="piston-v0",
    entry_point="envs.pistonball.pistonball_simple_new:parallel_env",
)