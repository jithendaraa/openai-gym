import numpy as np
import gym
from AC_continuous_model import Agent
from utils import plotLearning

agent = Agent(alpha=5e-6, beta=1e-5, input_dims=[2], gamma=0.99,
              layer1_size=256, layer2_size=256)

env = gym.make('MountainCarContinuous-v0')
score_history = []
num_episodes = 30

for i in range(num_episodes):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action = np.array(agent.choose_action(obs)).reshape((1,))
        obs_, reward, done, info = env.step(action)
        agent.learn(obs, reward, obs_, done)
        obs = obs_
        score += reward
        env.render()
    score_history.append(score)
    print('episode ', i, 'score %.2f' % score)
filename = 'mountaincar-continuous.png'
x = [i for i in range(num_episodes)]
plotLearning(x, score_history, score_history, filename)