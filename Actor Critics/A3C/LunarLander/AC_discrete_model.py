import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ActorCritiNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorCritiNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        state = torch.Tensor(obs).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return pi, v

class Agent(object):
    def __init__(self, alpha, input_dims, gamma=0.99, layer1_size=256,
                layer2_size=256, n_actions=4):
        self.gamma = gamma
        self.actor_critic = ActorCritiNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions)
        self.log_probs = None

    def choose_action(self, obs):
        policy, _ = self.actor_critic.forward(obs)
        policy = F.softmax(policy)
        action_probs = torch.distributions.Categorical(policy)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)

        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value = self.actor_critic.forward(state)
        _, critic_value_ = self.actor_critic.forward(state_)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_critic.device)
        delta = reward + self.gamma * critic_value_ * (1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = (delta) ** 2
        # print("Actor loss: ", actor_loss)
        # print("Critic Loss: ", critic_loss)
        # print("Total loss: ", actor_loss + critic_loss)
        (actor_loss+critic_loss).backward()
        self.actor_critic.optimizer.step()