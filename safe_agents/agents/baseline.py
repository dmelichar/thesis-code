from safe_agents.safety.heuristic import controller

class BaselineAgent:
    def __init__(self, env):
        self.env = env

    def train(self, episodes=1000, render=False):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        scores, safety = [], []
        for e in range(episodes):
            done = False
            score = 0
            state = self.env.reset()
            safe_ep = []

            while not done:
                # get action for the current state and go one step in environment
                action = controller(self.env, state)
                next_state, reward, done, info = self.env.step(action)
                safe_ep.append(info['status'])

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    print(f"episode: {e}  | score: {score}")
            safety.append(safe_ep)
        return scores, safety


if __name__ == '__main__':
    import gym
    env = gym.make('LunarSafe-v0')
    agent = BaselineAgent(env)
    scores, safety = agent.train(episodes=10, render=True)
    print(scores, safety)

