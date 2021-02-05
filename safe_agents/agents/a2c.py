import numpy as np

from safe_agents.nn.networks import actor_model
from safe_agents.nn.networks import critic_model


class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0] - 1  # minus for safety
        self.action_size = env.action_space.n

        self.discount_factor = 0.99

        # create model for policy network
        self.actor = actor_model(self.state_size, self.action_size)
        self.critic = critic_model(self.state_size, self.action_size)

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def update_network(self, state, action, reward, next_state, done):
        target = np.zeros((1, 1))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def train(self, episodes=300, render=False):
        scores, safety = [], []
        for e in range(episodes):
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state-1, [1, self.state_size])

            while not done:
                if render:
                    self.env.render(mode="human")

                action = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                safety.append(state[0][-1])
                next_state = np.reshape(next_state-1, [1, self.state_size])
                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done or score == 499 else -100

                self.update_network(state, action, reward, next_state, done)

                score += reward
                state = next_state

                if done:
                    # every episode, plot the play time
                    score = score if score == 500.0 else score + 100
                    scores.append(score)
                    print("episode:", e, "  score:", score)

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if np.mean(scores[-min(10, len(scores)) :]) > 490:
                        return

            if e % 50 == 0:
                pass
                # save the model on episode len

        return scores, safety

    def save(self, save_loc):
        if self.actor is not None and self.critic is not None:
            self.actor.save_weights(f"{save_loc}actor.h2")
            self.critic.save_weights(f"{save_loc}critic.h2")

    def load(self, save_loc):
        if self.actor is not None and self.critic is not None:
            self.actor.load_weights(f"{save_loc}actor.h2")
            self.critic.load_weights(f"{save_loc}critic.h2")


if __name__ == "__main__":
    from safe_agents.utils import plot_visuals
    import gym

    env = gym.make("LunarSafe-v0")

    agent = A2CAgent(env)
    scores, safety = agent.train(episodes=15)

    plot_visuals(agent, scores, safety)
