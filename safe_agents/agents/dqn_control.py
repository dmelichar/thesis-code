import numpy as np

from safe_agents.safety.heuristic import controller
from safe_agents.agents.dqn import DQNAgent


class DQNControlAgent(DQNAgent):
    def __init__(self, env):
        super(DQNControlAgent, self).__init__(env)

    def get_action(self, state):
        return super().get_action(state)

    def train(self, episodes=1000, render=False):
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

                self.remember(state, action, reward, next_state, done)
                self.update_network()

                score += reward
                state = next_state

                if done:
                    self.update_target_model()
                    scores.append(score)
                    safety.append(status)
                    safe = status.count(1)
                    unsafe = status.count(0)
                    risk_rate = 0 if unsafe == 0 else unsafe / (safe+unsafe)
                    print(
                        f"\tepisode: {e}  | "
                        f"score: {score}  | "
                        f"memory: {len(self.memory)} | "
                        f"epsilon: {self.epsilon} | "
                        f"risk_rate {risk_rate}"
                    )

        return scores, safety


if __name__ == "__main__":
    import gym

    env = gym.make("LunarSafe-v0")
    agent = DQNControlAgent(env)
    scores, safety = agent.train(episodes=10, render=False)
    print("======================")
    print(f"total_reward: {sum(scores)}")
    print(f"safe_s {sum(x.count(1) for x in safety)} | unsafe_s {sum(x.count(0) for x in safety)}")

