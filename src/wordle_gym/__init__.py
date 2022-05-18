from gym.envs import register

register(
    id='Wordle-v0',
    entry_point='src.wordle_gym.env_0:Wordle'
)

register(
    id='Wordle-v1',
    entry_point='src.wordle_gym.env_1:Wordle'
)