import tensorflow as tf
import gym
env = gym.make('CartPole-v0')
# Rewards you need to win in this environment
avg_win_reward = 195.0 # over 100 trials
rewards = []
ep_reward = 0
episodes = 5 # or trials
for i_episode in range(episodes):
    observation = env.reset()
    ep_reward = 0
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        ep_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    rewards.append(ep_reward)
env.close()
print(rewards)
avg_reward = sum(rewards)/episodes
if avg_reward < avg_win_reward:
    print("YOU LOSE")
else: 
    print("YOU WON")
print("Avg. reward: ", avg_reward)
print("You need an average reward of", avg_win_reward ,"over 100 trials to win")