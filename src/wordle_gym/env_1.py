import gym
import random
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Dict
from string import ascii_uppercase as alphabets

@dataclass
class EnvironmentConfig:
    N_ALPHABETS: int = 26
    N_ROUNDS: int = 6
    WORD_LENGTH: int = 5
    N_BOARD_STATES: int = 390
    STATE_LENGTH: int = 417
    GAMMA: float = 0.9
    corpus_path: str = "src/wordle_gym/corpus.txt"
    solution_path: str = "src/wordle_gym/possible_solutions.txt"
    state_path: str = "src/wordle_gym/state_dict_1.json"
    reward_dict: Dict[str, int] = field(default_factory=lambda: {
        'wrong letter': -1,
        'wrong position': 1,
        'correct position': 3})
    position_dict: Dict[str, int] = field(default_factory=lambda: {
        'wrong letter': 0,
        'wrong position': 1,
        'correct position': 2})

class Wordle(gym.Env):
    """
    Observation space: [S0, ... , S389, A0, ..., A25, R] -> (417,) shape V1
    """
    def __init__(self, mode: Literal['easy', 'medium', 'hard', 'extra']):
        self.config = EnvironmentConfig()
        if mode == 'easy':
            self.solutions = random.sample(
                self.get_words(self.config.solution_path), 100)
        elif mode == 'medium':
            self.solutions = random.sample(
                self.get_words(self.config.solution_path), 1000)
        elif mode == 'hard':
            self.solutions = self.get_words(self.config.solution_path)
        elif mode == 'extra':
            self.solutions = self.get_words(self.config.corpus_path)
        self.words = self.get_words(self.config.corpus_path)
        self.state_dict = self.get_state_dict(self.config.state_path)
        self.action_space = gym.spaces.Discrete(len(self.words))
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(self.config.STATE_LENGTH,), dtype=np.float64)

    def get_words(self, path):
        with open(path, "r") as f:
            words = [word.strip("\n").upper() for word in f]
        return words

    def get_state_dict(self, path):
        with open(path, 'r') as f:
            state_dict = json.load(f)
        return state_dict

    def get_info(self):
        return {
            "hidden_word": self.hidden_word,
            "guess_word": self.guess_word,
            "reward": self.rewards}

    def get_discount_rate(self):
        if self.game_state[-1] == 0:
            discount_rate = 1.0
        elif self.game_state[-1] == 1:
            discount_rate = self.config.GAMMA
        else:
            discount_rate = self.config.GAMMA ** (self.game_state[-1]+1)
        return discount_rate

    def get_state_index(self, letter, state, position):
        flag = self.config.position_dict[state]
        state_key = "|".join([letter, str(position), str(flag)])
        return self.state_dict[state_key]

    def reset(self):
        self.hidden_word = random.choice(self.solutions)
        self.observation_space = np.zeros(self.config.STATE_LENGTH,)
        self.game_state = np.zeros(self.config.STATE_LENGTH,)
        return self.observation_space

    def step(self, action):
        """
        Uses an additive structure for states
        """
        # Obtain literal guess word from action
        self.guess_word = self.words[action]

        self.rewards = 0
        discount_rate = self.get_discount_rate()
        done = False

        # Reset board states
        self.game_state[:self.config.N_BOARD_STATES] = np.zeros(self.config.N_BOARD_STATES)

        # Update game space and reward
        for pos, (i, j) in enumerate(zip(self.guess_word, self.hidden_word)):
            alphabet_id = alphabets.index(i)
            if i == j:
                state = "correct position"
                reward = self.config.reward_dict[state] * discount_rate
            elif i in self.hidden_word:
                state = "wrong position"
                reward = self.config.reward_dict[state] * discount_rate
            else:
                state = "wrong letter"
                reward = self.config.reward_dict[state] / discount_rate

            state_id = self.get_state_index(i, state, pos)
            self.game_state[state_id] = 1 # Updates board state cues that have been attempted in episode
            self.game_state[self.config.N_BOARD_STATES+alphabet_id] = 1 # Updates alphabets cues that have been attempted in episode
            self.rewards += reward

        self.game_state[-1] += 1

        if (self.game_state[-1] == self.config.N_ROUNDS) or (self.guess_word == self.hidden_word): # check if round = 6, end early
            done = True

        return self.game_state, self.rewards, done, self.get_info()

    def render(self):
        return

    def test(self, policy):
        time_step = self.reset()
        done = False
        while True:
            action = policy.action(time_step)
            current_state, reward, done, info = self.step(action)
            self.render()
            if done:
                print(current_state)
                break                        
        self.close()