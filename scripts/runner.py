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
        typer.echo(f"[info] Unsupported environment. Available {env_ids}")
        sys.exit(1)


@app.command()
def train(env_name: str, agent: str, episodes: int, save_loc: str = "./models/"):
    typer.echo(f"[info] Setting up environment {env_name}")
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    env = setup_env(env_name)

    agents = sa.agents.__all__
    if agent not in agents:
        typer.echo(f"[info] Unsupported agent. Available {agents}")
        sys.exit(1)
    elif agent == "DQN":
        agent = sa.agents.DQNAgent(env)
    elif agent == "DQNControl":
        agent = sa.agents.DQNControl(env)
    elif agent == "A2C":
        agent = sa.agents.A2CAgent(env)
    elif agent == "A2CControl":
        agent = sa.agents.A2CControl(env)
    elif agent == "Baseline":
        agent = sa.agents.Baseline(env)

    typer.echo(f"[info] {agent} training")
    scores, safety = agent.train(episodes=episodes)
    agent.save(save_loc=save_loc)
    typer.echo(f"[info] Agent saved to {save_loc}")
    sa.plot_visuals(agent, scores, safety)
    typer.echo(f"[info] Finished. Goodbye.")


@app.command()
def evaluate(env_name: str, agent: str, episodes: int, save_loc: str = "./models/"):
    typer.echo(f"[info] Setting up environment {env_name}")
    env = setup_env(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agents = sa.agents.__all__
    if agent not in agents:
        typer.echo(f"[info] Unsupported agent. Available {agents}")
        sys.exit(1)
    elif agent == "DQN":
        agent = sa.agents.DQNAgent(env)
        agent.load(save_loc=save_loc)
    elif agent == "DQNControl":
        agent = sa.agents.DQNControlAgent(env)
        agent.load(save_loc=save_loc)
    elif agent == "A2C":
        agent = sa.agents.A2CAgent(env)
        agent.load(save_loc=save_loc)
    elif agent == "A2CControl":
        agent = sa.agents.A2CControlAgent(env)
        agent.load(save_loc=save_loc)
    elif agent == "Baseline":
        agent = sa.agents.Baseline(env)

    typer.echo(f"[info] {agent} running")
    for e in range(episodes):
        done = False
        state = env.reset()
        while not done:
            env.render(mode="human")
            action = agent.get_action(state)
            state, _, done, _ = env.step(action)

    typer.echo(f"[info] Finished. Goodbye.")


if __name__ == "__main__":
    app()
