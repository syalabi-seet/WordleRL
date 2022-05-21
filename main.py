import os
from src.wordle_agent.dqn_agent import WordleAgent, AgentConfig

config = AgentConfig()
config.SEED = 42
config.ITERATIONS = int(2e5)
config.INITIAL_COLLECT_STEPS = int(1e3)
config.REPLAY_BUFFER_MAX_LENGTH = int(5e4)
config.BATCH_SIZE = 64
config.LEARNING_RATE = 1e-5
config.LOG_INTERVAL = int(1e3)
config.EVAL_EPISODES = int(1e3)
config.fc_layer_params = (1024, 1024, 1024)
config.EPSILON = 0.3
config.PREFETCH_BUFFER = 3
config.mode = "easy"
config.EPOCHS = config.ITERATIONS // config.LOG_INTERVAL
config.history_dir = os.path.join(
    config.dir, 'history', f"history_{config.mode}.json")

if __name__ == "__main__":
    agent = WordleAgent(config)
    agent.train()