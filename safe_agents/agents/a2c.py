import numpy as np
import random

from collections import deque

from safe_agents.nn.networks import a2c_actor
from safe_agents.nn.networks import a2c_critic


class A2CAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99

        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create model for policy network
        self.actor = a2c_actor(state_size, action_size)
        self.critic = a2c_critic(state_size, action_size)

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def remember(self, state, action, reward, next_state, done, bounds):
        self.memory.append((state, action, reward, next_state, done, bounds))

    # update policy network every episode
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

    def train_agent(self, episodes=1000, render=False):
        scores = []
        for e in range(episodes):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state[:-2], [1, self.state_size])

            while not done:
                if render:
                    self.env.render()

                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                bounds = np.reshape(
                    next_state[-2:], [1, 2]
                )  # seperate safety from other
                next_state = np.reshape(next_state[:-2], [1, self.state_size])
                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done or score == 499 else -100

                self.remember(state, action, reward, next_state, done, bounds)
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
        return scores

    # a bit hacky but ok
    def get_safe_bounds(self):
        return [(i[0][0][-1], i[0][0][-2]) for i in self.memory]

    def save(self, save_loc):
        self.actor.save_weights(f"{save_loc}a2c-actor.h2")
        self.critic.save_weights(f"{save_loc}a2c-critic.h2")

    def load(self, save_loc):
        self.actor.load_weights(f"{save_loc}a2c-actor.h2")
        self.critic.load_weights(f"{save_loc}a2c-critic.h2")
