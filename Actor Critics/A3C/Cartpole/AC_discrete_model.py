import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

'''
Agent has 2 networks: Actor and Critic
Actor: Policy approximator -> tells the agent what action to take
       This is basically the policy(π), which is approximated by a neural network N with
       policy parameters Θ that we need to optimise to Θ* for optimal policy π* using the 
       gradient of the log prob or grad of log policy (d/dΘ(log π(Θ))):
       ΔΘ = alpha * d/dΘ(log π(Θ))

Critic: Tells how much an action is good or bad. Look at the value [A(n) - b(n)] where A is actor and b is our critic
        defined as the expected sum of rewards(with discounting of γ per time step). But this is the same as the definition
        of our value function V(s)! Thus the critic is V(s). 
        And therefore, A(n) - b(n) becomes [R(t+1) + γ*V(S[n+1]) - V(S[n])], ie., the TD Error
'''

class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.lr = lr

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)


    def forward(self, state):
        state = torch.Tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Agent(object):
    '''
    For the Cartpole env: n_actions = 2
    alpha - lr for Actor
    beta - lr for Critic
    gamma - discount factor
    '''
    def __init__(self, alpha, beta, input_dims, gamma=0.99, l1_size=256, l2_size=256, n_actions=2):
        self.gamma = gamma
        self.log_probs = None # For actor network updation
        self.actor = GenericNetwork(alpha, input_dims, l1_size,
                                    l2_size, n_actions)
        self.critic = GenericNetwork(beta, input_dims, l1_size,
                                    l2_size, n_actions=1)
    
    def choose_action(self, state):
        # softmax ensures actions add up to one which is a requirement for probabilities
        probabilities = F.softmax(self.actor.forward(state))
        # create a distribution that is modelled on these probabilites
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        
        # action is a tensor, but open ai needs the int
        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(new_state)

        # TD Error: [R(t+1) + γ*V(S[n+1]) - V(S[n])]
        delta = ((reward + self.gamma*critic_value_*(1-int(done))) - critic_value)

        # Maximize self.log_probs * delta, greatest possible future reward
        actor_loss = -self.log_probs * delta
        critic_loss = (delta)**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()
