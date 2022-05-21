import enum
import json
import gym
import random
import regex as re
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Dict
from string import ascii_uppercase as alphabets

@dataclass
class EnvironmentConfig:
    SEED: int = 42
    N_ALPHABETS: int = 26
    N_ROUNDS: int = 6
    N_FLAGS: int = 3
    WORD_LENGTH: int = 5
    STATE_LENGTH: int = 32
    N_POSITION_STATES: int = 78
    GAMMA: float = 0.9
    corpus_path: str = "src/wordle_gym/corpus.txt"
    solution_path: str = "src/wordle_gym/possible_solutions.txt"
    state_path: str = "src/wordle_gym/state_dict.json"
    reward_map: Dict[str, int] = field(default_factory=lambda: {
        'win': 15,
        'wrong letter': -3,
        'wrong position': 2,
        'correct position': 3})
    position_map: Dict[str, int] = field(default_factory=lambda: {
        'neutral': 0,
        'wrong letter': 1,
        'wrong position': 2,
        'correct position': 3})

class Wordle(gym.Env):
    """
    Observation space: [A0, ..., A25, P1, ..., P5, R] -> (32,) shape V0
    """
    def __init__(self, mode: Literal['easy', 'medium', 'hard', 'extra']):
        self.config = EnvironmentConfig()
        if mode == 'easy':
            random.seed(self.config.SEED)
            self.solutions = random.sample(
                self.get_words(self.config.solution_path), 100)
        elif mode == 'medium':
            random.seed(self.config.SEED)
            self.solutions = random.sample(
                self.get_words(self.config.solution_path), 1000)
        elif mode == 'hard':
            self.solutions = self.get_words(self.config.solution_path)
        elif mode == 'extra':
            self.solutions = self.get_words(self.config.corpus_path)
        self.words = self.get_words(self.config.corpus_path)
        self.alphabets = alphabets
        self.statekey_dict = self.get_state_dict(self.config.state_path)
        self.statevalue_dict = {v: k for k, v in self.statekey_dict.items()}
        self.action_space = gym.spaces.Discrete(len(self.words))
        self.observation_space = self.get_observation_space()

    def get_observation_space(self):
        board_low = np.zeros(self.config.STATE_LENGTH, dtype=np.float32)
        board_high = np.ones(self.config.STATE_LENGTH, dtype=np.float32)

        return gym.spaces.Dict({
            'board': gym.spaces.Box(board_low, board_high, shape=(self.config.STATE_LENGTH,), dtype=np.float32),
            'mask': gym.spaces.Box(0, 1, shape=(len(self.words),), dtype=np.float32)})

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

    def decode_alphabet_state(self, state):      
        alphabet_state = state[:len(alphabets)]
        alphabet_state = np.where(alphabet_state==1.)[0]
        alphabet_state = set([self.alphabets[a] for a in alphabet_state])
        return alphabet_state

    def decode_position_state(self, state, valid = True):
        if valid:
            position_state = state[len(alphabets):-1] * self.config.N_POSITION_STATES
        else:
            position_state = state[len(alphabets):-1]
        position_state = [self.statevalue_dict[round(p)] for p in position_state]
        return position_state

    def encode_state(self, letter, state):
        flag = self.config.position_map[state]
        state_key = "|".join([letter, str(flag)])
        return self.statekey_dict[state_key]

    def update_mask(self, state):
        """
        Updates mask using rules curated from game round clues.
        Uses a subtractive logic to eliminate invalid candidates from word pool.
        """
        position_state = self.decode_position_state(state['board'], valid=False)
        mask_state = state['mask']
        valid_pattern = ""
        invalid_letters = "^"

        for board_element in position_state:
            letter, flag = board_element.split("|")
            if int(flag) == 2: # If wrong position
                valid_pattern += f"[^{letter}]"
            elif int(flag) == 3:
                valid_pattern += f"[{letter}]"
            elif int(flag) == 1:
                valid_pattern += f"[A-Z]"
                invalid_letters += letter

        valid_pattern = "^" + valid_pattern.replace("A-Z", invalid_letters) + "$"
        for word_id in np.where(mask_state==1.)[0]: # Iterate over all possible candidates
            word = self.words[word_id] # Obtain raw word
            valid_match = re.match(valid_pattern, word) # Capture words that are valid using regex
            if not valid_match: # if valid_match is None
                mask_state[word_id] = 0  
        return mask_state

    def reset(self):
        self.round = 0
        self.hidden_word = random.choice(self.solutions)
        self.observation_space = {
            'board': np.zeros(self.config.STATE_LENGTH, dtype=np.float32),
            'mask': np.ones(len(self.words), dtype=np.float32)}
        self.game_state = {
            'board': np.zeros(self.config.STATE_LENGTH, dtype=np.float32),
            'mask': np.ones(len(self.words), dtype=np.float32)}
        return self.observation_space

    def step(self, action):
        assert action in np.where(self.game_state['mask']==1.)[0], "Guess word is not in candidates list"

        # NOTE: Obtain literal guess word from action
        self.guess_word = self.words[action]

        # NOTE: Update game space and reward    
        self.rewards = 0
        discount_rate = self.config.GAMMA ** self.round
        done = False

        for pos, (i, j) in enumerate(zip(self.guess_word, self.hidden_word)):
            alphabet_id = alphabets.index(i)
            if i == j:
                state = "correct position"
                reward = self.config.reward_map[state] * discount_rate
            elif i in self.hidden_word:
                state = "wrong position"
                reward = self.config.reward_map[state] / discount_rate
            else:
                state = "wrong letter"
                reward = self.config.reward_map[state] / discount_rate

            self.game_state['board'][alphabet_id] = 1 # NOTE: Updates alphabets cues that have been attempted in episode
            self.game_state['board'][self.config.N_ALPHABETS+pos] = self.encode_state(i, state) # NOTE: Updates positional cues in episode
            self.rewards += reward

        self.round += 1

        if self.guess_word == self.hidden_word:
            self.rewards += self.config.reward_map['win'] * discount_rate
            done = True
        elif self.round == self.config.N_ROUNDS:
            done = True

        # Update mask with cues
        self.game_state['mask'] = self.update_mask(self.game_state)

        # Normalize
        self.game_state['board'][self.config.N_ALPHABETS:-1] = (
            self.game_state['board'][self.config.N_ALPHABETS:-1] / self.config.N_POSITION_STATES)
        self.game_state['board'][-1] = self.round / self.config.N_ROUNDS

        return self.game_state, self.rewards, done, self.get_info()

    def render(self, current_state):
        print("Board: ", current_state['board'])
        print("Mask:", np.sum(current_state['mask']))

    def test(self, policy):
        time_step = self.reset()
        done = False
        while True:
            print(policy)
            action = policy.action(time_step).action
            current_state, reward, done, info = self.step(action)
            self.render(current_state)
            if done:
                break                        
        self.close()