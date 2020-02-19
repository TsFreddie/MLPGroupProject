#import gym
import numpy as np
from building_env import BuildingEnv
from agent import Agent

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
agent = Agent(env.observation_space, env.action_space)

num_episodes = 1000
render = True

for i in range(num_episodes):
    state = env.reset()
    steps = 0
    rewards = 0
    episode_loss = 0

    while True:
        if render:
            env.render()
        
        action = agent.action(state)
        next_state, reward, done, info = env.step(action)

        agent.experience(state, action, reward, next_state, done)

        steps += 1
        episode_loss += agent.train(32)

        state = next_state

        rewards += reward

        if done:
            break
    
    avg_loss = episode_loss / steps
    print(f"Episode: {i}, rewards: {rewards}, steps: {steps}, avg_loss: {avg_loss}")