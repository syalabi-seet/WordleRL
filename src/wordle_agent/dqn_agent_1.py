import os
import json
import gym
import regex as re
import numpy as np
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
from loguru import logger
from multiprocessing import cpu_count

from src.wordle_gym.env_1 import Wordle

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
    WORKERS: int = cpu_count() // 4
    LEARNING_RATE: float = 1e-4
    LOG_INTERVAL: int = int(5e3)
    EVAL_EPISODES: int = int(1e3)
    fc_layer_params: tuple = (512, 512)
    EPSILON: float = 0.1
    mode: str = 'easy'
    dir: str = os.path.join('src', 'wordle_agent')
    ckpt_dir: str = os.path.join(dir, 'checkpoint')
    policy_dir: str = os.path.join(dir, 'policy')
    history_dir: str = os.path.join(dir, 'history', 'history.json')
    EPOCHS: int = ITERATIONS // LOG_INTERVAL

class ExperienceReplay:
    def __init__(self, agent, env, config):
        self.config = config
        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=env.batch_size,
            device='gpu:0',
            max_length=self.config.REPLAY_BUFFER_MAX_LENGTH)

        self.policy_state = agent.policy.get_initial_state(env.batch_size)
       
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=self.config.WORKERS, 
            sample_batch_size=self.config.BATCH_SIZE,
            single_deterministic_pass=False,
            num_steps=2)
        self.dataset = self.dataset.shuffle(self.config.BATCH_SIZE)
        self.dataset = self.dataset.prefetch(self.config.PREFETCH_BUFFER)

        self.iterator = iter(self.dataset)

    def initialize(self, env, policy, steps):
        for _ in tqdm(range(steps), total=steps, desc="Initial Experience"):
            self.update(env, policy)
            
    def update(self, env, policy):
        time_step = env.current_time_step()
        action_step = policy.action(time_step, self.policy_state)
        self.policy_state = action_step.state
        next_time_step = env.step(action_step.action)
        timestamp_trajectory = trajectory.from_transition(
            time_step, action_step, next_time_step)

        self.replay_buffer.add_batch(timestamp_trajectory)

class WordleAgent:
    def __init__(self, config):
        self.config = config
        self.env = gym.make("Wordle-v1", mode=self.config.mode)
        self.train_env = TFPyEnvironment(suite_gym.wrap_env(self.env))
        self.eval_env = TFPyEnvironment(suite_gym.wrap_env(self.env))
        self.step_counter = tf.Variable(0)
        self.agent = self.get_agent()
        self.experience_replay = ExperienceReplay(
            self.agent, self.train_env, self.config)

    def splitter_fn(self, observation):
        mask = tf.zeros(shape=(len(self.env.words),))
        try:
            observation_state = tf.squeeze(observation, axis=0).numpy()
            board_states = observation_state[:self.env.config.N_BOARD_STATES]
            alphabet_states = observation_state[self.env.config.N_BOARD_STATES:-1]
            print(board_states, alphabet_states)
            alphabets_used = []
            for alphabet_state in alphabet_states:
                print()

        except:
            print(observation)
        return observation, tf.expand_dims(mask, axis=0)

    def get_agent(self):
        q_net = QNetwork(
            self.train_env.observation_spec(),
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
        total_rounds = 0.
        for _ in tqdm(range(num_episodes), total=num_episodes, desc="Validation"):
            time_step = env.reset()

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = env.step(action_step.action)

            episode_return = time_step.reward.numpy()[0]
            round = time_step.observation.numpy()[0][-1]

            total_returns += episode_return
            total_rounds += round

        return {
            'avg_returns': total_returns / num_episodes,
            'avg_rounds': total_rounds / num_episodes}

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
                'iteration': [0], 'returns': [metrics['avg_returns']],
                'rounds': [metrics['avg_rounds']]}
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
        print(self.agent._q_network.summary())

    def train(self):
        self.initialize()

        for epoch in range(self.config.EPOCHS):
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            progbar = tf.keras.utils.Progbar(
                self.config.LOG_INTERVAL, interval=0.05,
                stateful_metrics=['loss', 'avg_returns', 'avg_rounds'])

            for step in range(1, self.config.LOG_INTERVAL+1):
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
                ('avg_rounds', metrics['avg_rounds'])],
                finalize=True)

            self.history['iteration'].append(step)
            self.history['returns'].append(metrics['avg_returns'])
            self.history['rounds'].append(metrics['avg_rounds'])
            
            with open(self.config.history_dir, "w") as f:
                json.dump(self.history, f, indent=4)

            self.train_checkpoint.save(global_step)

            if self.best_returns < metrics['avg_returns']:
                self.best_returns = metrics['avg_returns']
                self.policy_saver.save(self.config.policy_dir)
                logger.info("Policy improved")

            logger.info(f"Global step: {sum(self.history['iteration'])}")

    def test(self):
        policy = tf.saved_model.load(self.config.policy_dir)
        time_step = self.eval_env.reset()
        while True:
            action_step = policy.action(time_step)
            time_step = self.eval_env.step(action_step.action)
            print(time_step.observation.numpy()[0][26:-1])
            if time_step.observation.numpy()[0][-1] == 6:
                break

    def get_plots(self):
        with open(self.config.history_dir, "r") as f:
            history = json.load(f)
        plt.plot(history['returns'])