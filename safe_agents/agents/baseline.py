from safe_agents.safety.heuristic import controller


class BaselineAgent(object):
    def __init__(self, env):
        self.env = env

    def get_action(self, s):
        return controller(self.env, s)

    def train(self, episodes=1000, render=False):
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        scores, safety = [], []
        for e in range(episodes):
            done = False
            score = 0
            state = self.env.reset()
            status = []

            while not done:
                if render:
                    self.env.render(mode="human")

                action = self.get_action(state)
                state, reward, done, info = self.env.step(action)
                status.append(info["status"])
                score += reward

                if done:
                    safe = status.count(1)
                    unsafe = status.count(0)
                    risk_rate = 0 if unsafe == 0 else unsafe / (safe + unsafe)
                    scores.append(score)
                    safety.append(status)
                    print(f"\tepisode: {e}  | score: {score} | risk rate: {risk_rate}")

        return scores, safety

    def save(self, save_loc):
        pass

    def load(self, load_loc):
        pass

    def __str__(self):
        return self.__class__.__name__


if __name__ == "__main__":
    import gym

    env = gym.make("LunarSafe-v0")
    agent = BaselineAgent(env)
    scores, safety = agent.train(episodes=10, render=True)
    print("======================")
    print(f"total_reward: {sum(scores)}")
    print(
        f"safe_s {sum(x.count(1) for x in safety)} | unsafe_s {sum(x.count(0) for x in safety)}"
    )
