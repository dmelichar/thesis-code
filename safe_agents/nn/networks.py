from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

def dqn_model(state_size, action_size, learning_rate=0.0001):
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
    #model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


def ppo_actor(state_size, action_size, learning_rate=0.00025):
    initializer = initializers.RandomNormal(stddev=0.01)
    model = Sequential()
    model.add(Dense(512, input_dim=state_size, activation='relu', kernel_initializer=initializer))
    model.add(Dense(256, input_dim=state_size, activation='relu', kernel_initializer=initializer))
    model.add(Dense(64, input_dim=state_size, activation='relu', kernel_initializer=initializer))
    model.add(Dense(action_size, activation='softmax'))
    # model.summary()
    model.compile(loss=ppo_actor_loss, optimizer=Adam(lr=learning_rate))
    return model


def ppo_critic(state_size, action_size, learning_rate=0.00025):
    model = Sequential()
    model.add(Dense(512, input_dim=state_size, kernel_initializer='he_uniform'))
    model.add(Dense(256, input_dim=state_size, kernel_initializer='he_uniform'))
    model.add(Dense(64, input_dim=state_size, kernel_initializer='he_uniform'))
    model.add(Dense(1))
    # model.summary()
    model.compile(loss=ppo_critic_loss, optimizer=Adam(lr=learning_rate))
    return model

def ppo_actor_loss(y_true, y_pred):
    pass

def ppo_critic_loss(y_true, y_pred):
    pass

# approximate policy and value using Neural Network
# actor: state is input and probability of each action is output of model
def a2c_actor(state_size, action_size, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(512, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_size, activation='softmax', kernel_initializer='he_uniform'))
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    return model

# critic: state is input and value of state is output of model
def a2c_critic(state_size, action_size, learning_rate=0.005):
    model = Sequential()
    model.add(Dense(512, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(64, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
    #model.summary()
    model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return model

def sc_actor():
    pass

def sc_critic():
    pass

