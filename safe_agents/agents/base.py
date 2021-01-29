

class Agent(object):
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.model = None

    def save(self, filename):
        if self.model is not None:
            self.model.save_weights(filename)

    def load(self, filename):
        if self.model is not None:
            self.model.load_weights(filename)

    def update_network(self):
        pass

    def train_agent(self, episodes=1000, render=False):
        pass
