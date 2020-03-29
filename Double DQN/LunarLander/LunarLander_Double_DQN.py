import gym
import numpy as np
from model import DoubleDQNAgent
from utils import plotLearning

env = gym.make('LunarLander-v2')
num_games = 1500

agent = DoubleDQNAgent(alpha=5e-4,
              gamma=0.99,
              n_actions=4,
              epsilon=1.0,
              batch_size=64,
              input_dims=[8],
              eps_dec=1e-3,
              eps_end=0.01,
              mem_size=1000000)

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
        agent.remember(observation, action, reward, observation_, done)        
        observation = observation_
        agent.learn()
        env.render()

    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    print('episode ', i+1, 'score %.1f avg_score %.1f espilon %.2f' % \
        (score, avg_score, agent.epsilon))
    
x = [i+1 for i in range(num_games)]
# plotLearning(x, scores, eps_history, filename)
