import numpy as np
import gym
from AC_discrete_model import Agent
import matplotlib.pyplot as plt
from utils import plotLearning
from gym import wrappers

agent = Agent(alpha=1e-5, beta=5e-4, input_dims=4,
              gamma=0.99, n_actions=2, l1_size=32, l2_size=32)

env = gym.make('CartPole-v1')
score_history = []
n_episodes = 2500
for i in range(n_episodes):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        score += reward
        agent.learn(obs, reward, obs_, done)
        obs = obs_
        env.render()
    print("episode ", i, 'score %.3f' % score)
    score_history.append(score)

filename = 'cartpole.png'
x = [i+1 for i in range(n_episodes)]
plotLearning(x, score_history, score_history, filename)