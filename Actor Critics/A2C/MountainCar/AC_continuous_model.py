import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()

        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        state = torch.tensor(obs, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class Agent(object):
    # COntinuous mountain car problem
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2, layer1_size=64, layer2_size=64, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        
        self.actor = GenericNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=1)
    
    def choose_action(self, obs):
        mu, sigma = self.actor.forward(obs)
        sigma = torch.exp(sigma)
        action_probs = torch.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=torch.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        # Bounding from -1 to +1
        action = torch.tanh(probs)

        return action.item()

    # Learning is through temporal difference: learns at every time step using old state and new state and reward
    # Using replay buffer may improve performance

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value = self.critic.forward(new_state)
        critic_value_ = self.critic.forward(state)

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        delta = reward + self.gamma * critic_value_ * (1-int(done)) - critic_value

        actor_loss =  -self.log_probs * delta
        critic_loss = delta ** 2

        (actor_loss+critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        


        