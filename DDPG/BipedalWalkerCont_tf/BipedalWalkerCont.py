import gym
import numpy as np
from ddpg_model_tf import Agent
from utils import plotLearning
import time

env = gym.make('BipedalWalker-v3')

alpha = 0.0001
beta = 0.001
input_dims = [24]
tau = 0.001
n_actions = 4

agent = Agent(alpha, beta, input_dims, tau, env, n_actions=n_actions)
np.random.seed(0)

# agent.load_models()
num_episodes = 5000dsdsasdsadas

score_history = []

for i in range(num_episodes):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(obs)
        new_state, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        # slow render first and last 10 episodes
        # if i < 10 or i > num_episodes - 10:
        #     time.sleep(0.003)
        env.render()

    score_history.append(score)
    print('episode: ', i, 'score: %.2f', score, 
        'trailing 100 games avg: %.2f' % np.mean(score_history[-100:]))
     
    if i % 25 == 0:
        agent.save_models()
    
filename = 'bipedal.png'
plotLearning(score_history, filename, window=100)
