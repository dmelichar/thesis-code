import gym
import safe_agents as sa
import typer
import inspect
from pathlib import Path
from typing import List

app = typer.Typer()
available = [i[0] for i in inspect.getmembers(sa.agents, inspect.isclass)]


def setup(a, env):
    if a == "A2CAgent":
        return sa.agents.A2CAgent(env)
    elif a == "A2CControlAgent":
        return sa.agents.A2CControlAgent(env)
    elif a == "A2CSafeAgent":
        return sa.agents.A2CSafeAgent(env)
    elif a == "BaselineAgent":
        return sa.agents.BaselineAgent(env)
    elif a == "DQNAgent":
        return sa.agents.DQNAgent(env)
    elif a == "DQNControlAgent":
        return sa.agents.DQNControlAgent(env)
    elif a == "DQNSafeAgent":
        return sa.agents.DQNSafeAgent(env)


@app.command()
def train(
    agent: List[str] = typer.Option(..., help=f"Any of {str(available)}"),
    episodes: int = typer.Option(..., help="Number of training episodes"),
    env_name: str = typer.Option("LunarSafe-v0", help="Environment to run."),
    save_loc: str = typer.Option("./models/", help="Path to save to after training"),
    plot_loc: str = typer.Option("./results/", help="Path to save plots to"),
    compare: bool = typer.Option(False, help="Generate comparission plots"),
):
    if any(a not in available for a in agent):
        typer.echo(f"[info] Unsupported agent. Available {str(available)}")
        raise typer.Abort()

    typer.echo(f"[info] Training {str(agent)} on environment {env_name}")
    env = gym.make(env_name)
    agents = [setup(a, env) for a in agent]
    if compare:
        typer.echo(f"[info] Also setting up BaselineAgent for comparission")
        agents.append(setup("BaselineAgent", env))
    data = []
    for a in agents:
        typer.echo(f"[info] Agent {a} started")
        scores, safety = a.train(episodes=episodes)
        data.append(
            {
                "agent": str(a),
                "safe": [ep.count(1) for ep in safety],
                "unsafe": [ep.count(0) for ep in safety],
                "scores": scores,
            }
        )
        Path(save_loc).mkdir(parents=True, exist_ok=True)
        a.save(save_loc=save_loc)
        typer.echo(f"[info] Agent {a} saved to {save_loc}")
        sa.utils.plot_visuals(a, scores, safety, loc=plot_loc)
        typer.echo(f"[info] Agent {a} plots saved to {plot_loc}")
    if compare:
        typer.echo(
            f"[info] Comparing performance of {str(agent)} against BaselineAgent"
        )
        sa.utils.plot_comparisson(data, episodes, loc=plot_loc)
        typer.echo(f"[info] Comparission saved to {plot_loc}")
    typer.echo(f"[info] Finished. Goodbye.")


@app.command()
def evaluate(
    agent: List[str] = typer.Option(..., help=f"Any of {str(available)}"),
    episodes: int = typer.Option(..., help="Number of training episodes"),
    env_name: str = typer.Option("LunarSafe-v0", help="Environment to run."),
    save_loc: str = typer.Option("./models/", help="Path to save to after training"),
    compare: bool = False,
):
    if any(a not in available for a in agent):
        typer.echo(f"[info] Unsupported agent. Available {str(available)}")
        raise typer.Abort()

    typer.echo(f"[info] Evaluating {str(agent)} on environment {env_name}")
    env = gym.make(env_name)
    agents = [setup(a, env) for a in agent]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    for a in agents:
        typer.echo(f"[info] Agent {a} started")
        for e in range(episodes):
            done = False
            state = env.reset()
            while not done:
                env.render(mode="human")
                action = a.get_action(state)
                state, _, done, _ = env.step(action)
    typer.echo(f"[info] Finished. Goodbye.")


if __name__ == "__main__":
    app()
