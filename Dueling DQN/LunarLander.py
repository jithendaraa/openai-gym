import gym
import numpy as np
from model import Agent
from utils import plotLearning
import time

env = gym.make('LunarLander-v2')
num_games = 1000
load_checkpoint = False

print(env.action_space.n, *env.observation_space.shape)

agent = Agent(gamma=0.99,
              epsilon=1.0,
              alpha=5e-4,
              n_actions=4,
              input_dims=[8],
              mem_size=1000000,
              batch_size=64,
              eps_min=0.01,
              eps_dec=1e-3,
              replace=100)

if load_checkpoint:
    agent.load_models()

filename = 'lunar_lander_dueling.png'

scores = []
eps_history = []


for i in range(num_games):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, int(done))
        agent.learn()
        observation = observation_
        env.render()

    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('episode ', i+1, 'score %.1f avg_score %.1f espilon %.2f' % \
        (score, avg_score, agent.epsilon))
    eps_history.append(agent.epsilon)

x = [i+1 for i in range(num_games)]
plotLearning(x, scores, eps_history, filename)
