import numpy as np

from safe_agents.safety.heuristic import controller
from safe_agents.agents.dqn import DQNAgent


class DQNControlAgent(DQNAgent):
    def __init__(self, env):
        super(DQNControlAgent, self).__init__(env)

    def train(self, episodes=1000, render=False):
        scores, safety = [], []
        for e in range(episodes):
            done, override = False, False
            score = 0
            safe_ep = []
            state = self.env.reset()

            while not done:
                if render:
                    self.env.render(mode="human")

                if override:
                    action = controller(self.env, state)
                else:
                    action = self.get_action(state)
                
                next_state, reward, done, info = self.env.step(action)
                safe_ep.append(info['status'])

                self.remember(state, action, reward, next_state, done)

                self.update_network()
                score += reward
                state = next_state

                if done:
                    # every episode update the target model to be same with model
                    self.update_target_model()
                    scores.append(score)
                    print(
                        f"episode: {e}  | "
                        f"score: {score}  | "
                        f"memory: {len(self.memory)} | "
                        f"epsilon: {self.epsilon}"
                    )

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if np.mean(scores[-min(10, len(scores)) :]) > 490:
                        return
            safety.append(safe_ep)
        return scores, safety

if __name__ == '__main__':
    import gym
    env = gym.make('LunarSafe-v0')
    agent = DQNAgent(env)
    scores, safety = agent.train(episodes=15)
    print(scores)
    print()
    print(safety)
    print()
