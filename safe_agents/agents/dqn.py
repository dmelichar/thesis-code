import numpy as np
import random

from collections import deque

from safe_agents.nn.networks import dqn_model


class DQNAgent:
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.train_start = 1000

        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = dqn_model(state_size, action_size)
        self.target_model = dqn_model(state_size, action_size)

        # initialize target model
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def remember(self, state, action, reward, next_state, done, bounds):
        self.memory.append((state, action, reward, next_state, done, bounds))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_network(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i])
                )

        self.model.fit(
            update_input, target, batch_size=self.batch_size, epochs=1, verbose=0
        )

    def train_agent(self, episodes=1000, render=False):
        scores = []
        for e in range(episodes):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state[:-2], [1, self.state_size])

            while not done:
                if render:
                    self.env.render(mode="human")

                # get action for the current state and go one step in environment
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                bounds = np.reshape(
                    next_state[-2:], [1, 2]
                )  # seperate safety from other
                next_state = np.reshape(next_state[:-2], [1, self.state_size])

                self.remember(state, action, reward, next_state, done, bounds)

                self.update_network()
                score += reward
                state = next_state

                if done:
                    # every episode update the target model to be same with model
                    self.update_target_model()
                    # every episode, plot the play time
                    score = score if score == 500 else score + 100
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
        return scores

    # a bit hacky but ok
    def get_safe_bounds(self):
        return [(i[0][0][-1], i[0][0][-2]) for i in self.memory]

    def save(self, save_loc):
        if self.model is not None:
            self.model.save_weights(f"{save_loc}dqn.h2")

    def load(self, save_loc):
        if self.model is not None:
            self.model.load_weights(f"{save_loc}dqn.h2")
