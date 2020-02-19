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
        
        self.observation_size = np.product(observation_space.shape)
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
        model.add(keras.layers.Dense(30, input_dim=self.observation_size, activation='relu'))
        model.add(keras.layers.Dense(30, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        
        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss='mse')

        return model

    def save(self, name):
        self.model.save(name)

    def experience(self, state, action, reward, next_state, done):
        state = state.flatten()
        next_state = next_state.flatten()
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
            return 0

        samples = random.sample(self.memory, batch)

        states = np.array([val[0] for val in samples])
        actions = np.array([val[1] for val in samples])
        rewards = np.array([val[2] for val in samples])
        next_states = np.array([val[3] for val in samples])
        valid = np.array([not val[4] for val in samples])
        indexes = np.arange(batch)

        target_qvs = self.model(states).numpy()
        next_qvs = self.model(next_states).numpy()

        rewards[valid] += self.gamma * np.amax(next_qvs[valid, :], axis=1)
        
        target_qvs[indexes, actions] = rewards

        history = self.model.train_on_batch(states, target_qvs)
        
        if (self.eps >= self.min_eps):
            self.eps *= self.eps_decay
        
        return history


