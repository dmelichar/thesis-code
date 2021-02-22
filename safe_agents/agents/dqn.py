import numpy as np
import random
import datetime

from collections import deque

from safe_agents.nn.networks import dqn_model

class DQNAgent(object):
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.train_start = 1000

        # create replay memory using deque
        self.memory = deque(maxlen=10000)

        # create main model and target model
        self.model = dqn_model(self.state_size, self.action_size)
        self.target_model = dqn_model(self.state_size, self.action_size)

        # initialize target model
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(np.reshape(state, [1, self.state_size]))
            return np.argmax(q_value[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
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

    def train(self, episodes=1000, render=False):
        scores, safety = [], []
        for e in range(episodes):
            done = False
            score = 0
            status = []
            state = self.env.reset()

            while not done:
                if render:
                    self.env.render(mode="human")

                # get action for the current state and go one step in environment
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                status.append(info["status"])

                self.remember(state, action, reward, next_state, done)

                self.update_network()
                score += reward
                state = next_state

                if done:
                    # every episode update the target model to be same with model
                    self.update_target_model()
                    scores.append(score)
                    safety.append(status)
                    safe = status.count(1)
                    unsafe = status.count(0)
                    risk_rate = 0 if unsafe == 0 else (unsafe / (safe+unsafe))
                    print(
                        f"\tepisode: {e}  | "
                        f"score: {score}  | "
                        f"memory: {len(self.memory)} | "
                        f"epsilon: {self.epsilon} | "
                        f"risk_rate: {risk_rate} "
                    )
        return scores, safety

    def save(self, save_loc):

        if self.model is not None:
            self.model.save_weights(f"{save_loc}{str(self)}.h2")

    def load(self, load_loc):
        if self.model is not None:
            self.model.load_weights(f"{save_loc}{str(self)}.h2")

    def __str__(self):
        return self.__class__.__name__


if __name__ == "__main__":
    import gym

    env = gym.make("LunarSafe-v0")
    #agent = DQNAgent(env)
    #scores, safety = agent.train(episodes=10, render=False)
    print("======================")
    print(f"total_reward: {sum(scores)}")
    print(f"safe_s {sum(x.count(1) for x in safety)} | unsafe_s {sum(x.count(0) for x in safety)}")
