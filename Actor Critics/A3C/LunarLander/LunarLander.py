import numpy as np
import gym
from AC_discrete_model import Agent
import matplotlib.pyplot as plt
from utils import plotLearning
from gym import wrappers

agent = Agent(alpha=1e-5, input_dims=[8], gamma=0.99, n_actions=4, layer1_size=2048, layer2_size=512)

env = gym.make('LunarLander-v2')
num_episodes = 10
score_history = []

for i in range(num_episodes):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        agent.learn(obs, reward, obs_, done)
        obs = obs_
        score += reward

    score_history.append(score)
    print('episode ', i, 'score %.2f' % score)

filename = 'lunar_lander_discrete.png'
x = [i for i in range(num_episodes)]
plotLearning(x, score_history, score_history, filename)

