# This gets rid of NumPy FutureWarnings that occur at TF import
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# This gets rid of TF 2.0 related deprecation warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
def train(env_name: str, agent: str, episodes: int, save_loc: str = "./models/"):
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    env = setup_env(env_name)

    agents = sa.agents.__all__
    if agent not in agents:
        typer.echo(f"Unsupported agent. Available {agents}")
        sys.exit(1)
    elif agent == "DQN":
        agent = sa.agents.DQNAgent(env)
    elif agent == "A2C":
        agent = sa.agents.A2CAgent(env)
    elif agent == "SafetyController":
        agent = sa.agents.SafetyController(env)

    scores, safety = agent.train(episodes=episodes)
    agent.save(save_loc=save_loc)
    typer.echo(f"[info] Agent saved to {save_loc}")
    sa.plot_visuals(agent, scores, safety)


@app.command()
def evaluate(env_name: str, agent: str, episodes: int, save_loc: str = "./models/"):
    env = setup_env(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agents = sa.agents.__all__
    if agent not in agents:
        typer.echo(f"Unsupported agent. Available {agents}")
        sys.exit(1)
    elif agent == "DQN":
        agent = sa.agents.DQNAgent(env)
    elif agent == "A2C":
        agent = sa.agents.A2CAgent(env)
    elif agent == "SafetyController":
        agent = sa.agents.SafetyController(env)

    agent.load(save_loc=save_loc)

    for e in range(episodes):
        done = False
        score = 0
        state = env.reset()
        while not done:
            env.render(mode="human")
            action = agent.get_action(state)
            env.step(action)


if __name__ == "__main__":
    app()
