import os
import json
import gym
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger
import multiprocessing as mp

from src.wordle_gym.env import Wordle

import tensorflow as tf

from tf_agents.environments import suite_gym, TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DdqnAgent

from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory

from tf_agents.utils import common
from tf_agents.policies import PolicySaver

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@dataclass
class AgentConfig:
    SEED: int = 42
    ITERATIONS: int = int(1e7)
    INITIAL_COLLECT_STEPS: int = int(1e3)
    COLLECT_STEPS_PER_ITERATION: int = 1
    REPLAY_BUFFER_MAX_LENGTH: int = int(1e7)
    BATCH_SIZE: int = 64
    PREFETCH_BUFFER: int = 3
    WORKERS: int = mp.cpu_count() // 4
    LEARNING_RATE: float = 1e-4
    LOG_INTERVAL: int = int(5e3)
    EVAL_EPISODES: int = int(1e3)
    fc_layer_params: tuple = (512, 512)
    EPSILON: float = 0.1
    mode: str = 'easy'
    dir: str = os.path.join('src', 'wordle_agent')
    ckpt_dir: str = os.path.join(dir, 'checkpoint')
    policy_dir: str = os.path.join(dir, 'policy')
    history_dir: str = os.path.join(dir, 'history', f"history_{mode}.json")
    EPOCHS: int = ITERATIONS // LOG_INTERVAL

class ExperienceReplay:
    """
    Refresh replay buffer with newer experiences
    """
    def __init__(self, agent, env, config):
        self.config = config
        self.env = env
        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=env.batch_size,
            device='cpu:*',
            dataset_drop_remainder=True,
            max_length=self.config.REPLAY_BUFFER_MAX_LENGTH)
      
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=self.config.WORKERS, 
            sample_batch_size=self.config.BATCH_SIZE,
            single_deterministic_pass=False,
            num_steps=2)

        self.iterator = iter(self.dataset)

    def initialize(self, env, policy, steps):
        for _ in tqdm(range(steps), total=steps, desc="Initial Experience"):
            self.update(env, policy)
            
    def update(self, env, policy):
        time_step = env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = env.step(action_step.action)
        timestamp_trajectory = trajectory.from_transition(
            time_step, action_step, next_time_step)

        # NOTE: To only add positive samples
        # reward = next_time_step.reward.numpy()[0]
        # if reward > 0.0:
        self.replay_buffer.add_batch(timestamp_trajectory)

class WordleAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make("Wordle-v0", mode=self.config.mode)
        self.train_env = TFPyEnvironment(suite_gym.wrap_env(self.env))
        self.eval_env = TFPyEnvironment(suite_gym.wrap_env(self.env))
        self.step_counter = tf.Variable(0)
        self.agent = self.get_agent()        
        self.experience_replay = ExperienceReplay(
            self.agent, self.train_env, self.config)

    def splitter_fn(self, observations):
        return observations['board'], observations['mask']

    def get_agent(self):
        q_net = QNetwork(
            self.train_env.observation_spec()['board'],
            self.train_env.action_spec(),
            fc_layer_params=self.config.fc_layer_params)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.LEARNING_RATE)

        return DdqnAgent(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=self.config.EPSILON,
            optimizer=optimizer,
            observation_and_action_constraint_splitter=self.splitter_fn,
            train_step_counter=self.step_counter)

    def compute_metrics(self, env, policy, num_episodes):
        total_returns = 0.
        total_rounds = 0
        total_wins = 0
        total_correct_positions = 0
        total_candidates = 0
        for _ in tqdm(range(num_episodes), total=num_episodes, desc='Validation'):
            time_step = env.reset()

            wins = 0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)
            episode_return = time_step.reward.numpy()[0]
            observation_state = time_step.observation['board'].numpy()[0]
            mask_state = time_step.observation['mask'].numpy()[0]
            game_state = self.env.decode_position_state(observation_state, valid=True)
            game_state = [int(s.split("|")[1]) for s in game_state]
            correct_positions = len([i for i in game_state if i == 3]) / 5
            round_state = round(observation_state[-1] * 6)
                    
            if correct_positions == 1.:
                wins += 1

            total_returns += episode_return
            total_rounds += round_state
            total_wins += wins
            total_correct_positions += correct_positions
            total_candidates += np.sum(mask_state, axis=0)

        return {
            'avg_returns': total_returns / num_episodes,
            'avg_rounds': total_rounds / num_episodes,
            'avg_wins': total_wins / num_episodes,
            'avg_correct_pos': total_correct_positions / num_episodes,
            'avg_candidates': total_candidates / num_episodes}

    def initialize(self):
        if os.path.exists(self.config.history_dir):
            with open(self.config.history_dir, "r") as f:
                self.history = json.load(f)
            self.best_returns = max(self.history['returns'])
            logger.info(f"Training resumed from step {sum(self.history['iteration'])}")
        else:
            self.agent.initialize()
            self.time_step = self.train_env.reset()
            self.experience_replay.initialize(
                self.train_env, self.agent.policy, self.config.INITIAL_COLLECT_STEPS)
            metrics = self.compute_metrics(
                self.eval_env, self.agent.policy, self.config.EVAL_EPISODES)
            self.history = {
                'iteration': [0], 
                'returns': [round(metrics['avg_returns'], 4)],
                'rounds': [metrics['avg_rounds']], 
                'wins': [metrics['avg_wins']],
                'correct_positions': [round(metrics['avg_correct_pos'],4)],
                'candidates': [round(metrics['avg_candidates'], 4)]}
            self.best_returns = -np.inf

        self.train_checkpoint = common.Checkpointer(
            ckpt_dir=self.config.ckpt_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.experience_replay.replay_buffer,
            global_step=self.step_counter)
        
        self.policy_saver = PolicySaver(self.agent.policy)        

        self.agent.train = common.function(self.agent.train)
        self.train_checkpoint.initialize_or_restore()
        self.agent._q_network.summary()

    def train(self):
        self.initialize()

        for epoch in range(self.config.EPOCHS):
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            progbar = tf.keras.utils.Progbar(
                self.config.LOG_INTERVAL, interval=0.05,
                stateful_metrics=['loss', 'avg_returns', 'avg_rounds', 'avg_wins', 'avg_correct_pos', 'avg_candidates'])

            for step in range(1, self.config.LOG_INTERVAL+1):
                # Reset states after every episode
                self.train_env.reset()

                for _ in range(self.config.COLLECT_STEPS_PER_ITERATION):
                    self.experience_replay.update(self.train_env, self.agent.collect_policy)
                
                experience, _ = next(self.experience_replay.iterator)
                train_loss = self.agent.train(experience).loss
                global_step = self.agent.train_step_counter.numpy()
                progbar.update(step, values=[('loss', train_loss)], finalize=False)

            metrics = self.compute_metrics(
                self.eval_env, self.agent.policy, self.config.EVAL_EPISODES)
            progbar.update(step, values=[
                ('loss', train_loss), ('avg_returns', metrics['avg_returns']),
                ('avg_rounds', metrics['avg_rounds']), ('avg_wins', metrics['avg_wins']),
                ('avg_correct_pos', metrics['avg_correct_pos']), ('avg_candidates', metrics['avg_candidates'])],
                finalize=True)

            self.history['iteration'].append(step)
            self.history['returns'].append(round(metrics['avg_returns'],4))
            self.history['rounds'].append(metrics['avg_rounds'])
            self.history['wins'].append(metrics['avg_wins'])
            self.history['correct_positions'].append(round(metrics['avg_correct_pos'],4))
            self.history['candidates'].append(round(metrics['avg_candidates'],4))
            
            self.train_checkpoint.save(global_step)

            if self.best_returns < metrics['avg_returns']:
                self.best_returns = metrics['avg_returns']
                self.policy_saver.save(self.config.policy_dir)
                logger.info("Policy improved")

            with open(self.config.history_dir, "w") as f:
                json.dump(self.history, f, indent=4)

            logger.info(f"Global step: {sum(self.history['iteration'])}")
            logger.info(f"Buffer size: {self.experience_replay.replay_buffer.num_frames().numpy()}")

    def play(self):
        policy = tf.saved_model.load(self.config.policy_dir)
        time_step = self.eval_env.reset()
        while True:
            action_step = policy.action(time_step)
            time_step = self.eval_env.step(action_step.action)
            observation_space = time_step.observation['board'].numpy()[0]
            game_state = self.env.decode_position_state(observation_space, valid=True)
            game_state = [int(s.split("|")[1]) for s in game_state]
            round_state = round(observation_space[-1] * 6)
            print(f"Round: {round_state}, Board state: {game_state}")
            if (round_state == 6) or (game_state == [3] * 5):
                break

    def get_plots(self):
        with open(self.config.history_dir, "r") as f:
            history = json.load(f)
        plt.plot(history['returns'])