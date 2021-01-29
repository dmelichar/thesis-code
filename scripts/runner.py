# This gets rid of NumPy FutureWarnings that occur at TF import
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
# This gets rid of TF 2.0 related deprecation warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Disable other useless logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

import gym
from gym import envs
import safe_agents as sa
import numpy as np
import typer
from pathlib import Path
import sys

app = typer.Typer()

def setup_env(env):
    all_envs = envs.registry.all()
    env_ids = [spec.id for spec in all_envs]
    if env in env_ids:
        return gym.make(env)
    else:
        typer.echo(f"Unsupported environment. Available {env_ids}")
        sys.exit(1)

@app.command()
def train(env_name: str, agent: str, episodes: int, save_loc: str = './models/'):
    env = setup_env(env_name)
    state_size = env.observation_space.shape[0] - 2
    action_size = env.action_space.n

    agents = sa.agents.__all__
    if agent not in agents:
        typer.echo(f"Unsupported agent. Available {agents}")
        sys.exit(1)
    elif agent == 'DQN':
        agent = sa.agents.DQNAgent(env, state_size, action_size)
    elif agent == 'PPO':
        agent = sa.agents.PPOAgent(env, state_size, action_size)
    elif agent == 'SafetyCritic':
        agent = sa.agents.SafetyCriticAgent(env, state_size, action_size)

    Path(save_loc).mkdir(parents=True, exist_ok=True)
    scores = agent.train_agent()
    agent.save(save_loc + 'dqn.h2')
    sa.plot_visuals(agent, scores)

@app.command()
def evaluate(env_name: str, agent: str, episodes: int, save_loc: str = './models/'):
    env = setup_env(env_name)
    state_size = env.observation_space.shape[0] - 2
    action_size = env.action_space.n

    agents = sa.agents.__all__
    if agent not in agents:
        typer.echo(f"Unsupported agent. Available {agents}")
        sys.exit(1)
    elif agent == 'DQN':
        agent = sa.DQNAgent(env, state_size, action_size)
    elif agent == 'PPO':
        agent = sa.PPOAgent(env, state_size, action_size)
    elif agent == 'SafetyCritic':
        agent = sa.SafetyCriticAgent(env, state_size, action_size)

    Path(save_loc).mkdir(parents=True, exist_ok=True)
    agent.load(save_loc + 'dqn.h2')

    for e in range(episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state[:-2], [1, state_size])
        while not done:
            env.render(mode='human')
            action = agent.get_action(state)
            obs, reward, done, info = env.step(action)

if __name__ == "__main__":
    app()

