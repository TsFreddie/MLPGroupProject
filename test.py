from grid_building import GridBuildingEnv
import numpy as np
import time
import os

import matplotlib
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from result_plotter import load_results, ts2xy, plot_results, X_TIMESTEPS
from stable_baselines import PPO2

log_dir = "./log/"
os.makedirs(log_dir, exist_ok=True)

building = np.array(
  [
    #  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
      [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], # 0
      [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], # 1
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 2
      [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], # 3
      [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], # 4
      [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], # 5
      [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1], # 6
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 7
      [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], # 8
      [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1], # 9
      [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1], # A
  ], dtype=np.int8
)

# building = np.array([
#   [0, 1, 0],
#   [0, 0, 0],
#   [0, 1, 0]
# ])

starts = [(0x2,0x2), (0x7,0x0), (0x7,0x7), (0x2,0x6), (0x2,0xC), (0x0,0x9), (0x7,0xC)]
exits = [(0x0,0x0), (0x0,0x3), (0x0,0xB), (0xA,0xE), (0xA,0x8), (0xA,0x0)]

# starts = [(2,2)]
# exits = [(0,0)]
env = GridBuildingEnv(building=building, starting_points=starts, exits=exits)
env = Monitor(env, log_dir, allow_early_resets=True)


def debug():
  obs, reward, done, info = env.step([0, 3])
  env.render()
  print(reward)

  while True:
    env.render()
    time.sleep(1)

# env = Monitor(env, log_dir, allow_early_resets=True)
# env = gym.make("CartPole-v4")

# best_mean_reward, n_steps = -np.inf, 0

# def callback(_locals, _globals):
#     """
#     Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
#     :param _locals: (dict)
#     :param _globals: (dict)
#     """
#     global n_steps, best_mean_reward
#     # Print stats every 1000 calls
#     # if (n_steps + 1) % 1000 == 0:
#     # Evaluate policy training performance
#     x, y = ts2xy(load_results(log_dir), 'timesteps')
#     if len(x) > 0:
#         mean_reward = np.mean(y[-128:])
#         print(x[-1], 'timesteps')
#         print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

        
      
#         result = plot_results([log_dir], )
        # New best model, you could save the agent here
        # if mean_reward > best_mean_reward:
        #     best_mean_reward = mean_reward
        #     # Example for saving best model
        #     print("Saving new best model")
        #     _locals['self'].save(log_dir + 'best_model.pkl')
    # n_steps += 1
    # Returning False will stop training early
#     return True

def train_run(steps=10000, save="model"):  
  model = PPO2(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=steps)

  result = plot_results([log_dir], steps, X_TIMESTEPS, "Exit Route")
  
  model.save('model')
  plt.savefig('train_2_plot.pdf')
  # obs = env.reset()
  # total_rewards = 0
  # for i in range(1000):
  #     action, _states = model.predict(obs)
  #     obs, reward, done, info = env.step(action)
  #     # total_rewards += reward
  #     print(f"Rewards: {reward}, Steps: {i}")
  #     # if (done):
  #     #     break
  #     env.render()
  # env.close()

train_run(15000)