from ddpg_tf import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('Pendulum-v0')
agent = Agent(alpha=1e-4, beta=1e-3, input_dims=[3], tau=0.001, env=env,
            batch_size=64, layer1_size=400, layer2_size=300, n_actions=1)

np.random.seed(0)
score_history = []
num_epsiodes = 2500

for i in range(num_epsiodes):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, obs_, done)
        agent.learn()
        score += reward
        obs = obs_
        env.render()
    score_history.append(score)
    print('episode ', i, 'score %.2f' % score,
                '100 game average %.2f' % np.mean(score_history[-100:]))

filename = 'pendulum.png'
x = [i for i in range(num_epsiodes)]
plotLearning(x, score_history, score_history, filename)