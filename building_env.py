import gym
import numpy as np
from gym import spaces

DIRS = [
    np.array([0, -1]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([-1, 0]),
]

eps = 1e-5

# class Dummy():
#     def __init__(self, r, c):
#         self.r = r
#         self.c = c
#     def move(self, direction):
#         pass
class BuildingEnv(gym.Env):
    # np array [n*m*2] - [smoke value]
    def __init__(self, building):
        super(BuildingEnv, self).__init__()

        self.action_space = spaces.Discrete(4) # orientation
        # self.lower_bound = np.ones(building.shape) * np.array([0])
        # self.higher_bound = np.ones(building.shape) * np.array([1, 3])
        # print(self.lower_bound)
        self.observation_space = spaces.Box(low=0, high=1, shape=building.shape, dtype=np.float16)

        self.dummy_bound = np.array(building.shape)
        self.building = building

    def reset(self):
        self.grid = self.building
        self.current_step = 0
        self.max_step = 100
        self.id = 0
        self.reward = 0
        
        self.dummies = np.array([0, 0])
        return self.grid
    
    def step(self, action):
        # action:
        # 0 - north
        # 1 - east
        # 2 - south
        # 3 - west
        
        self.reward -= 1
        self.current_step += 1
        self.dummies = self.dummies + DIRS[action] * 0.5
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
        
        self.reward -= self.grid[int(index[0])][int(index[1])]
        if (np.array_equal(index, np.array(self.grid.shape) - np.ones(2))):
            self.reward += 20
        
        return self.grid, self.reward, self.current_step >= self.max_step, {}

    def render(self, mode='human', close=False):
        print(self.current_step)
        print(self.dummies)
        

    