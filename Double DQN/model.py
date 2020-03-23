# Double DQN

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print(torch.cuda.is_available())

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, terminal


class DoubleDQN(nn.Module):
    def __init__(self, ALPHA, n_actions, input_dims):
        super(DoubleDQN, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.Q = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        Q = self.Q(l2)

        return Q

class DoubleDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                input_dims, eps_dec=0.996, eps_end=0.01, mem_size=1000000, replace_target=100):

        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.eps_dec = eps_dec
        self.eps_min = eps_end
        self.action_space = [i for i in range(self.n_actions)]
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, True)
        self.replace_target = replace_target
        self.q_eval = DoubleDQN(alpha, n_actions, input_dims)
        self.q_target = DoubleDQN(alpha, n_actions, input_dims)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(state).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()

        return action
    
    def learn(self):
        # What happens if we havent filled up batchsize of our memory
        if self.memory.mem_cntr < self.batch_size:
            return
    
        self.q_eval.zero_grad()
        self.q_target.zero_grad()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            
        state = torch.tensor(state).to(self.q_eval.device)
        new_state = torch.tensor(new_state).to(self.q_eval.device)
        reward = torch.tensor(reward).type(torch.FloatTensor).to(self.q_eval.device)
        done = torch.tensor(done).type(torch.FloatTensor).to(self.q_eval.device)

        # TO convert one hot action to normal ie [0,1,0,0] -> 1
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        
        q_next = self.q_target.forward(new_state).to(self.q_eval.device)        
        q_pred = self.q_eval.forward(state).to(self.q_eval.device)
        
        _, max_actions = torch.max(q_next, dim=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target = self.q_eval.forward(state).to(self.q_eval.device)
        q_target[batch_index, action_indices] = reward + self.gamma * q_next[batch_index, max_actions] * done

        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()
        self.decrement_epsilon()
