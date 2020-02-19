from building_env import BuildingEnv
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

building = np.array([
    [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0],
    [5.,5.,5.,5.,5.,5.,5.,5.,5.,.0],
    [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0],
    [.0,5.,5.,5.,5.,5.,5.,5.,5.,5.],
    [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0],
    [5.,5.,5.,5.,5.,5.,5.,5.,5.,.0],
    [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0],
    [.0,5.,5.,5.,5.,5.,5.,5.,5.,5.],
    [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0],
    [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0],
])

env = BuildingEnv(building)
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)

obs = env.reset()
total_rewards = 0
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_rewards += reward
    print(f"Rewards: {total_rewards}, Steps: {i}")
    if (done):
        break
    env.render()
env.close()
