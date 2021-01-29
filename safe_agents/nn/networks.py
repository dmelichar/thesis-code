from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def dqn_model(state_size, action_size, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
    #model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


def ppo_actor(state_size, action_size, learning_rate=0.001):
    initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    model = Sequential()
    model.add(Dense(512, input_dim=state_size, activation='relu', kernel_initializer=initializer))
    model.add(Dense(256, input_dim=state_size, activation='relu', kernel_initializer=initializer))
    model.add(Dense(64, input_dim=state_size, activation='relu', kernel_initializer=initializer))
    model.add(Dense(action_size, activation='softmax))
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    # alternative: loss as https://arxiv.org/abs/1707.06347
    return model


def ppo_critic(state_size, action_size, learning_rate=0.001):
    pass

def sc_actor():
    pass

def sc_critic():
    pass

