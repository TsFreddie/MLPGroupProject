import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import gym.spaces as spaces
from collections import deque
import random

class Agent:
    def __init__(self,
                 observation_space,
                 action_space,
                 learning_rate = 0.001,
                 gamma = 0.95,
                 max_eps = 1,
                 min_eps = 0.01,
                 eps_decay = 0.995,
                 saved_model = None):
        
        self.observation_shape = observation_space.shape
        self.action_space = action_space

        if (isinstance(action_space, spaces.Discrete)):
            self.action_size = action_space.n
        elif (isinstance(action_space, spaces.Box)):
            self.action_size = np.product(action_space.shape)
        
        self.lr = learning_rate
        self.gamma = gamma

        self.memory = deque(maxlen=2000)
        self.eps = max_eps
        self.min_eps = min_eps
        self.eps_decay = eps_decay

        if (saved_model is not None and os.path.isfile(saved_model)):
            self.model = keras.models.load_model(saved_model)
            print("Loaded Model:")
        else:
            self.model = self.Model()
            print("Creating new Model:")

        print(self.model.summary())

    def Model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(50, input_shape=self.observation_shape, activation='relu'))
        model.add(keras.layers.Dense(50, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss='mse')

        return model

    def save(self, name):
        self.model.save(name)

    def experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# def choose_action(state, primary_network, eps):
#     if random.random() < eps:
#         return random.randint(0, num_actions - 1)
#     else:
#         return np.argmax(primary_network(state.reshape(1, -1)))

    def _discrete_action(self, state):
        if np.random.rand() < self.eps:
            return self.action_space.sample()
        else:
            return np.argmax(self.model.predict(state.reshape(1, -1))[0])

    def action(self, state):
        return self._discrete_action(state)
    
    def train(self, batch):
        if (batch >= len(self.memory)):
            return None

        samples = random.sample(self.memory, batch)

        for state, action, reward, next_state, done in samples:
            state_flatten = state.reshape(1, -1)
            state_qvs = self.model.predict(state_flatten)

            if done:
                state_qvs[0][action] = reward
            else:
                next_qvs = self.model.predict(next_state.reshape(1, -1))
                state_qvs[0][action] = reward + self.gamma * np.amax(next_qvs)

            history = self.model.fit(x=state_flatten, y=state_qvs, verbose=0, epochs=1)
        
        if (self.eps >= self.min_eps):
            self.eps *= self.eps_decay
        
        return history


