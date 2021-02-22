import numpy as np

from safe_agents.safety.heuristic import controller
from safe_agents.agents.a2c import A2CAgent


class A2CControlAgent(A2CAgent):
    def __init__(self, env):
        super(A2CControlAgent, self).__init__(env)

    def get_action(self, state):
        return super().get_action(state)

    def train(self, episodes=300, render=False):
        scores, safety = [], []
        for e in range(episodes):
            done, override = False, False
            score = 0
            status = []
            state = self.env.reset()

            while not done:
                if render:
                    self.env.render(mode="human")

                if override:
                    action = controller(self.env, state)
                else:
                    action = self.get_action(state)

                next_state, reward, done, info = self.env.step(action)
                override = False if info["status"] else True  # safe = 1, unsafe = 0
                status.append(info["status"])

                self.update_network(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    safety.append(status)
                    safe = status.count(1)
                    unsafe = status.count(0)
                    risk_rate = 0 if unsafe == 0 else unsafe / (safe+unsafe)
                    print(
                        f"episode: {e}  | "
                        f"score: {score}  | "
                        f"risk_rate: {risk_rate} "
                    )
        return scores, safety


if __name__ == "__main__":
    import gym

    env = gym.make("LunarSafe-v0")
    agent = A2CControlAgent(env)
    scores, safety = agent.train(episodes=10, render=True)
    print("======================")
    print(f"total_reward: {sum(scores)}")
    print(f"safe_s {sum(x.count(1) for x in safety)} | unsafe_s {sum(x.count(0) for x in safety)}")
