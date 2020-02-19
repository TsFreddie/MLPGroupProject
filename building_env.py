import gym
import numpy as np
from gym import spaces
import tkinter as tk

DIRS = [
    np.array([0, -1]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([-1, 0]),
]

eps = 1e-5

UNIT = 40   # pixels

class BuildingEnv(gym.Env):
    # np array [n*m*2] - [smoke value, population]
    def __init__(self, building):
        super(BuildingEnv, self).__init__()

        self.action_space = spaces.Discrete(4) # orientation
        # self.lower_bound = np.ones(building.shape) * np.array([0])
        # self.higher_bound = np.ones(building.shape) * np.array([1, 3])
        # print(self.lower_bound)
        self.observation_space = spaces.Box(low=0, high=1, shape=(np.product(building.shape) * 2 + 2, ), dtype=np.float32)

        self.dummy_bound = np.array(building.shape)
        self.building = building

    # def _create_render(self):
    #     height = self.building.shape[0]
    #     width = self.building.shape[1]

    #     self.canvas = tk.Canvas(self, bg='white',
    #                        height=height * UNIT,
    #                        width=width * UNIT)

    #     # create grids
    #     for c in range(0, width * UNIT, UNIT):
    #         x0, y0, x1, y1 = c, 0, c, width * UNIT
    #         self.canvas.create_line(x0, y0, x1, y1)
    #     for r in range(0, height * UNIT, UNIT):
    #         x0, y0, x1, y1 = 0, r, width * UNIT, r
    #         self.canvas.create_line(x0, y0, x1, y1)

    def reset(self):
        self.grid = self.building
        self.history = np.zeros(self.building.shape)
        self.current_step = 0
        self.max_step = 200
        self.id = 0
        self.reward = 0
        
        self.dummies = np.array([0, 0])
        return np.concatenate((self.grid.flatten(), self.history.flatten(), self.dummies))
    
    def step(self, action):
        # action:
        # 0 - north
        # 1 - east
        # 2 - south
        # 3 - west
        
        # self.reward -= 1
        reward = -0.05
        self.current_step += 1

        done = self.current_step >= self.max_step

        self.dummies = self.dummies + DIRS[action]
        # if (not (tmp[0] < 0 or tmp[1] < 0 or tmp[0] > self.dummy_bound[0] or tmp[1] > self.dummy_bound[1])):
        #     self.dummies = tmp
        if (self.dummies[0] < 0):
            self.dummies[0] = 0
        if (self.dummies[1] < 0):
            self.dummies[1] = 0
        if (self.dummies[0] >= self.dummy_bound[0]):
            self.dummies[0] = self.dummy_bound[0] - eps
        if (self.dummies[1] >= self.dummy_bound[1]):
            self.dummies[1] = self.dummy_bound[1] - eps   

        index = np.floor(self.dummies)
        
        reward -= (self.grid[int(index[0])][int(index[1])] + self.history[int(index[0])][int(index[1])] * 0.01)
        self.history[int(index[0])][int(index[1])] += 1
        if (np.array_equal(index, np.array(self.grid.shape) - np.ones(2))):
            done = True
            reward += 1
        
        return np.concatenate((self.grid.flatten(), self.history.flatten(), self.dummies)), reward, done, {}

    def render(self, mode='human'):
        pass
        

    