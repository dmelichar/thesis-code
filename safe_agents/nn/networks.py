from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers

# fmt: off
def dqn_model(state_size, action_size, learning_rate=0.0001):
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(32, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(action_size, activation="linear", kernel_initializer="he_uniform"))
    # model.summary()
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model


def actor_model(state_size, action_size, learning_rate=0.001):
    actor = Sequential()
    actor.add(Dense(24, input_dim=state_size, activation="relu", kernel_initializer="he_uniform"))
    actor.add(Dense(action_size, activation="softmax", kernel_initializer="he_uniform"))
    #actor.summary()
    actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate))
    return actor


def critic_model(state_size, action_size, learning_rate=0.005):
    critic = Sequential()
    critic.add(Dense(24, input_dim=state_size, activation="relu", kernel_initializer="he_uniform"))
    critic.add(Dense(1, activation="linear", kernel_initializer="he_uniform"))
    #critic.summary()
    critic.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return critic

# fmt: on
