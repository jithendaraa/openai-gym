import gym
from model import DQNetwork, Agent
from utils import plotLearning
import numpy as np

if __name__ == "__main__":
    env_name = 'SpaceInvaders-v0'
    env = gym.make(env_name)

    print("Observation space size: ", env.observation_space.shape)
    print("Action space size: ", env.action_space)

    # brain = Agent(gamma=0.95, epsilon=1.0, alpha=0.03, maxMemorySize=5000, replace=10000)
    
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            # print("action:", action)
            # 0 -> shoot, 7 -> go right, 6 -> do nothing, 
            observation_, reward, done, info = env.step(action)
            env.render()
            if done and info['lives'] == 0:
                reward -= 100
            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), action, reward,
                                  np.mean(observation_[15:200, 30:125], axis=2))
            observation = observation_
        print('done initialising memory')

        scores = []
        epsHistory = []
        numGames = 5
        batch_size = 32

        for i in range(numGames):
            print("starting game", i+1, 'epsilon: %4f' % brain.EPSILON)
            epsHistory.append(brain.EPSILON)
            done = False
            observation = env.reset()
            frames = [np.sum(observation[15:200, 30:125], axis=2)]
            score = 0
            lastAction = 0
            # 3 frames

            while not done:
                if len(frames) == 3:
                    action = brain.chooseAction(frames)
                    frames = []
                else:
                    action = lastAction
                print(action)

                observation_, reward, done, info = env.step(0)
                score += reward
                frames.append(np.sum(observation[15:200, 30:125], axis=2))
                if done and info['lives'] == 0:
                    reward -= 100
                brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2), action, reward,
                                      np.mean(observation_[15:200, 30:125], axis=2))
                observation = observation_
                brain.learn(batch_size)
                lastAction = action
                env.render()
            scores.append(score)
            print('score: ', score)
            x = [i + 1 for i in range(numGames)]
            fileName = 'test' + str(numGames) + '.png'
            plotLearning(x, scores, epsHistory, fileName)