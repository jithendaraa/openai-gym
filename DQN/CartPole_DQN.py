import gym
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

env_name = 'CartPole-v0'
env = gym.make(env_name)
state = env.reset()
print(type(torch.from_numpy(state)), torch.from_numpy(state))

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
state_dim = env.observation_space.shape
action_size = env.action_space.n
state_tensor = torch.from_numpy(state)

# CartPole DQN
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_size):
        super().__init__()
        
        self.action_in = 0
        self.action_one_hot = F.one_hot(torch.arange(0,2)[self.action_in], num_classes=action_size).float()
        self.hidden1 = nn.Linear(*state_dim, 100) # Fully connected
        self.q_state = nn.Linear(100, action_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.q_state(x)
        return x

    def update_model(self, state, action, q_target):
        
        state_tensor_in = torch.from_numpy(state)
        self.action_in = action
        self.action_one_hot = F.one_hot(torch.arange(0,2)[self.action_in], num_classes=action_size)
        
        action_one_hot = F.one_hot(torch.arange(0,2)[action], num_classes=action_size).float()
        self.q_target_tensor_in = q_target
        q_state = self(state_tensor_in.float())
        q_state_action = torch.dot(q_state, action_one_hot).sum()
        
        EPOCHS = 1
        
        for epoch in range(EPOCHS):
            self.zero_grad()
            loss = F.mse_loss(q_state_action, q_target)
            loss.backward()
            self.optimizer.step()
            # print("LOSS:", loss)
        
    def get_q_state(self, state):
        state_tensor = torch.from_numpy(state)
        q_state = self(state_tensor.float())
        return q_state
        
class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.gamma = 0.97
        self.eps = 1.0
        
    def get_action(self, state):
        q_state = self.q_network.get_q_state(state)
        action_value, action_index = torch.max(q_state, 0)
        action_greedy = action_index.item()
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action
    
    def train(self, state, action, next_state, reward, done):
        q_next_state = self.q_network.get_q_state(next_state)
        q_next_state = (1-done) * q_next_state
        q_target = reward + self.gamma * q_next_state
        self.q_network.update_model(state, action, q_target)
        
        if done:
            self.eps = 0.99 * self.eps
        
        
agent = DQNAgent(env)
num_episodes = 850
max_reward = 0
avg_score = 0

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done)
        env.render()
        total_reward += reward
        state = next_state
    print("Episode: {}, total_reward: {:2f}".format(ep, total_reward))
    if(total_reward > max_reward):
        max_reward = total_reward
    if(ep >= 600):
        avg_score += total_reward

print(max_reward)
print("AVG SCORE:", avg_score/250)

# rewards = []
# print(rewards)
# avg_reward = sum(rewards)/num_episodes
# if avg_reward < 195:
#     print("YOU LOSE")
# else: 
#     print("YOU WON")
# print("Avg. reward: ", avg_reward)
# print("You need an average reward of", 195 ,"over 100 trials to win")