import numpy as np

from safe_agents.safety.heuristic import controller
from safe_agents.agents.a2c import A2CAgent

class A2CControlAgent(A2CAgent):
    def __init__(self, env):
        super(A2CControlAgent, self).__init__(env)

    def train(self, episodes=300, render=False):
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
                
                self.update_network(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    print("episode:", e, "  score:", score)

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if np.mean(scores[-min(10, len(scores)) :]) > 490:
                        return
            safety.append(safe_ep)
            if e % 50 == 0:
                pass
                # save the model on episode len
        
        return scores, safety


if __name__ == "__main__":
    from safe_agents.utils import plot_visuals
    import gym

    env = gym.make("LunarSafe-v0")

    agent = A2CAgent(env)
    scores, safety = agent.train(episodes=15)

    plot_visuals(agent, scores, safety)
