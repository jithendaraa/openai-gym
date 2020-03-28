from ddpg_model_pytorch import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make('LunarLanderContinuous-v2')
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[8], tau=0.001, env=env,
              batch_size=64, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []
num_episodes = 1000
for i in range(num_episodes):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action = agent.choose_action(obs)
        state_, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, state_, int(done))
        agent.learn()
        score += reward
        obs = state_
        env.render()

    score_history.append(score)
    print('episode ', i, 'score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))
    if i % 25 == 0:
        agent.save_models()

filename = 'lunar-lander.png'
x = [i for i in range(num_episodes)]
plotLearning(x, score_history, score_history, filename)