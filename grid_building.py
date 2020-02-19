import numpy as np
import gym
from gym import spaces
import pygame

MAX_WIDTH = 1000
MAX_HEIGHT = 600
MAX_BLOCK_SIZE = 40
BLACK = 0,0,0
GREEN = 0,255,0
BLUE = 0,0,255
RED = 255,0,0
WHITE = 255,255,255
ARROWS = [
    np.array(((-3, 15), (3, 15), (3, -5), (9, -5), (0, -15), (-9, -5), (-3, -5))),  # 0 UP
    np.array(((-15, -3), (-15, 3), (5, 3), (5, 9), (15, 0), (5, -9), (5, -3))),     # 1 RIGHT
    np.array(((-3, -15), (3, -15), (3, 5), (9, 5), (0, 15), (-9, 5), (-3, 5))),     # 2 DOWN
    np.array(((15, -3), (15, 3), (-5, 3), (-5, 9), (-15, 0), (-5, -9), (-5, -3))),  # 3 LEFT
]

class GridBuildingEnv(gym.Env):
    def __init__(
            self,
            # width = None,
            # height = None,
            building = None,
            starting_points = None,
            exits = None):
        
        super(GridBuildingEnv, self).__init__()

        if (building is not None):
            self.building = building
            self.width = building.shape[1]
            self.height = building.shape[0]
        # elif (width is not None and height is not None):
        #     self.width = width
        #     self.height = height
        #     self.building = np.zeros((height, width), dtype=np.int8)
        else:
            raise ValueError("Invalid GridBuildingStructure")
        
        self.starting_points = [] if starting_points is None else starting_points
        self.exits = [] if exits is None else exits

        self.signs_loc = []
        self.nsign = 0

        # auto assign sign location
        for i in range(self.height):
            for j in range(self.width):
                if (self.building[i][j] != 0):
                    continue
                cross = 0
                if (i > 0 and self.building[i-1][j] == 0):
                    cross += 1
                if (i < self.height - 1 and self.building[i+1][j] == 0):
                    cross += 1
                if (j < self.width - 1 and self.building[i][j+1] == 0):
                    cross += 1
                if (j > 0 and self.building[i][j-1] == 0):
                    cross += 1
                if (cross > 2):
                    self.signs_loc.append((i,j))
                    self.nsign += 1

        self.observation_space = spaces.Box(low=0, high=1, shape=(np.product(self.building.shape), ))
        self.action_space = spaces.MultiDiscrete([4] * self.nsign)

        self.reset()
        self.visual = {}

    def reset(self):
        self.sensor = np.zeros(self.building.shape)
        self.signs_value = np.zeros(self.nsign, dtype=int)

        self.total_reward = 0
        return self._get_obs()

    def setState(self, r, c, state = 1):
        self.building[r][c] = state
    
    def addStartingPoint(self, point):
        self.starting_points.append(point)
    
    def addExit(self, point):
        self.exits.append(point)
    
    def _pg_create_window(self):
        if not pygame.get_init():
            pygame.init()

        # Calculate apt canvas size
        self.visual["block_size"] = MAX_BLOCK_SIZE
        if (self.width * self.visual["block_size"] > MAX_WIDTH):
            self.visual["block_size"] = MAX_WIDTH // self.width
        if (self.height * self.visual["block_size"] > MAX_HEIGHT):
            self.visual["block_size"] = MAX_HEIGHT // self.height     
        
        canvas_width = self.width * self.visual["block_size"]
        canvas_height = self.height * self.visual["block_size"] 

        if ("root" in self.visual):
            pygame.quit()

        self.visual["root"] = pygame.display.set_mode((canvas_width, canvas_height))

        pygame.display.set_caption(f"Building: {self.width} x {self.height}")

    def _get_obs(self):
        obs = np.concatenate([self.sensor.flatten()])
        return obs

    def flood(self, point, path):
        if (point[0] < 0 or point[1] < 0 or point[0] >= self.height or point[1] >= self.width):
            return 0

        if (self.building[point[0]][point[1]] == 1):
            return 0

        if (point in path):
            return 0

        path.add(point)

        if (point in self.signs_loc):
            direction = self.signs_value[self.signs_loc.index(point)]
            if (direction == 0):
                return -1 + self.flood((point[0] - 1, point[1]), path)
            elif (direction == 1):
                return -1 + self.flood((point[0], point[1] + 1), path)
            elif (direction == 2):
                return -1 + self.flood((point[0] + 1, point[1]), path)
            elif (direction == 3):
                return -1 + self.flood((point[0], point[1] - 1), path)
        
        if (point in self.exits):
            return 500
    
        return (
            -1 + 
            self.flood((point[0] - 1, point[1]), path) + 
            self.flood((point[0], point[1] + 1), path) +
            self.flood((point[0] + 1, point[1]), path) +
            self.flood((point[0], point[1] - 1), path)
        )

    def step(self, action):
        self.signs_value = action
        
        rewards = 0
        for point in self.signs_loc:
            path = set()
            rewards += self.flood(point, path)

        self.total_reward += rewards

        done = False
        if (self.total_reward > self.width + self.height * len(self.starting_points) * 10):
            done = True

        return self._get_obs(), rewards, done, {}

    def render(self):
        if ("root" not in self.visual):
            self._pg_create_window()

        screen = self.visual["root"]
        size = self.visual["block_size"]
        rect_size = int(size * 0.8)
        dot_size = int(size * 0.7 // 2)

        screen.fill((255,255,255))

        for i in range(self.height):
            for j in range(self.width):      
                if (self.building[i][j] == 0):
                    pygame.draw.rect(screen, BLACK, (j * size, i * size, size, size), 1)
                else:
                    pygame.draw.rect(screen, BLACK, (j * size, i * size, size, size), 0)

        for point in self.exits:
            pygame.draw.rect(screen, GREEN, (point[1] * size + 4, point[0] * size + 4, rect_size, rect_size), 0)

        # for point in self.starting_points:
        #     pygame.draw.circle(screen, BLUE,
        #             (point[1] * size + size // 2, point[0] * size + size // 2), dot_size, 0)

        for i in range(len(self.signs_loc)):
            point = self.signs_loc[i]
            pygame.draw.polygon(screen, RED, ARROWS[self.signs_value[i]]
                 + np.array((point[1] * size + size // 2, point[0] * size + size // 2)))

        pygame.display.flip()

        for event in pygame.event.get():
            pass

    def close(self):
        if ("root" in self.visual):
            pygame.quit()
        self.visual.clear()


    
