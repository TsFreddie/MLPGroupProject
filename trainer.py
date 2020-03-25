import argparse
parser = argparse.ArgumentParser(description='MLP Project')
parser.add_argument('-r', '--run', action='store_true', default=False, dest='run')
parser.add_argument('-s', '--steps', dest='steps', type=int, default='5000')
parser.add_argument('-m', '--model', dest='model', type=str)
args = parser.parse_args()

if __name__ == "__main__":
  from grid_building import GridBuildingEnv
  import numpy as np
  import time
  import os
  import sys
  import datetime

  log_dir = "./log/"
  os.makedirs(log_dir, exist_ok=True)
  model_dir = "./models/"
  os.makedirs(model_dir, exist_ok=True)

  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines.bench import Monitor
  from stable_baselines.results_plotter import load_results, ts2xy
  from stable_baselines import PPO2, A2C
  from stable_baselines.common import set_global_seeds
  from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

  building = np.array(
    [
      #  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], # 0
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], # 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 2
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], # 3
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], # 4
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], # 5
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], # 6
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # 7
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1], # 8
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1], # 9
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1], # A
    ], dtype=np.int8
  )

  exits = [(0x0,0x0), (0x0,0x3), (0x0,0xB), (0xA,0xE), (0xA,0x8), (0xA,0x0)]

  model_name = (args.model or 'model-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
  # model_filename = 'model'
  model_filename = model_dir + model_name + '.pkl'
  
  if (args.run):
    env = GridBuildingEnv(building=building, exits=exits)
    model = PPO2.load(model_filename)
    obs = env.reset()
    total_rewards = 0
    steps = 0
    while True:
      action, _states = model.predict(obs)
      obs, reward, done, info = env.step(action)
      env.render()
      steps += 1
      total_rewards += reward
      print(f"Rewards: {reward}, Steps: {steps}", end='')
      if input().strip() == 'q':
          break

      if (done):
        print(f'total_rewards: {total_rewards}')
        env.reset()
        steps = 0
        total_rewards = 0

    input("Press enter to exit.")
    env.close()
  else:
    log_dir = "./log/%s/" % model_name
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    num_cpu = 6

    def make_env(rank, seed=0):
        def _init():
            env = GridBuildingEnv(building=building, exits=exits)
            env.seed(seed + rank)
            env = Monitor(env, log_dir + "%s[%d]" % (timestamp, rank), allow_early_resets=True)
            return env
        set_global_seeds(seed)
        return _init

    # def make_env(env_id):
    #   return lambda : Monitor(GridBuildingEnv(building=building, exits=exits), log_dir + "%s[%d]" % (timestamp, env_id), allow_early_resets=True)
    env = DummyVecEnv([make_env(i) for i in range(num_cpu)])
    if (os.path.isfile(model_filename)):
      model = PPO2.load(model_filename, env=env, verbose=1)
    else:
      model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(model_filename)

  