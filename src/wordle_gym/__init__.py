from gym.envs import register

register(
    id='Wordle-v0',
    entry_point='src.wordle_gym.env:Wordle'
)