import numpy as np

from safe_agents.nn.networks import ppo_actor
from safe_agents.nn.networks import ppo_critic


class PPOAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.99
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []

        self.actor = ppo_actor(state_size, action_size)
        self.critic = ppo_critic(state_size, action_size)

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype("float32") - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def get_action(self, state):
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    def discount_rewards(self, rewards):
        gamma = 0.99  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= np.std(discounted_r) + 1e-8  # divide by standard deviation
        return discounted_r

    def train_agent(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def save(self, save_loc):
        self.actor.save_weights(f"{save_loc}ppo-actor.h2")
        self.critic.save_weights(f"{save_loc}ppo-critic.h2")

    def load(self, save_loc):
        self.actor.load_weights(f"{save_loc}ppo-actor.h2")
        self.critic.load_weights(f"{save_loc}ppo-critic.h2")
