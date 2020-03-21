# Box2D
# needs python version <= 3.5
# conda create -n py35 python=3.5
# then do conda install -c https://conda.anaconda.org/kne pybox2d to get Box2d env up
# mujoco and rob works in base env

# DUELING DQN

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print(torch.cuda.is_available())

class ReplayBuffer(object):
  def __init__(self, mem_size, input_shape, n_actions):
    self.mem_size = mem_size
    self.mem_cntr = 0
    self.state_memory = np.zeros((self.mem_size, *input_shape), 
                                  dtype=np.float32)
    self.new_state_memory = np.zeros((self.mem_size, *input_shape), 
                                  dtype=np.float32)
    self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

  def store_transition(self, state, action, reward, next_state, done):
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.new_state_memory[index] = next_state
    self.terminal_memory[index] = done

    self.mem_cntr += 1

  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size)
    batch = np.random.choice(max_mem, batch_size, replace=False)

    states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    new_states = self.new_state_memory[batch]
    dones = self.terminal_memory[batch]

    return states, actions, rewards, new_states, dones

# Dueling DQN

class DuelingLinearDeepNetwork(nn.Module):
  def __init__(self, ALPHA, n_actions, name, input_dims, chkpt_dir='tmp/dueling_dqn'):
    super(DuelingLinearDeepNetwork, self).__init__()

    self.fc1 = nn.Linear(*input_dims, 128)
    self.fc2 = nn.Linear(128, 128)
    # In dueling DQN, instead of finding out just Q function, we calculate a value function and an advantage function
    self.V = nn.Linear(128, 1)
    self.A = nn.Linear(128, n_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
    self.loss = nn.MSELoss()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)
    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'dueling_dqn')

  def forward(self, state):
    l1 = F.relu(self.fc1(state))
    l2 = F.relu(self.fc2(l1))
    V = self.V(l2)
    A =self.A(l2)

    return V, A

  def save_checkpoint(self):
    print('.....saving checkpoint.....')
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    print('....loading checkpoint....')
    self.load_state_dict(torch.load(self.checkpoint_file))


class Agent(object):
  def __init__(self, gamma, epsilon, alpha, n_actions, input_dims, mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, replace=10000, chkpt_dir='tmp/dueling_dqn'):
    self.gamma = gamma
    self.epsilon = epsilon
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.batch_size = batch_size
    self.mem_size = mem_size
    self.action_space = [i for i in range(n_actions)]
    # Keep track of how many times we use the learning function to know when to update target net params
    self.learn_step_counter = 0
    self.replace_target_cnt = replace

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
    self.q_eval = DuelingLinearDeepNetwork(alpha, n_actions, 
                                           input_dims=input_dims,
                                           name='q_eval', 
                                           chkpt_dir=chkpt_dir)
    self.q_next_eval = DuelingLinearDeepNetwork(alpha, n_actions, 
                                           input_dims=input_dims,
                                           name='q_next_eval', 
                                           chkpt_dir=chkpt_dir)
  
  def store_transition(self, state, action, reward, next_state, done):
    self.memory.store_transition(state, action, reward, next_state, done)

  def choose_action(self, observation):
    if np.random.random() > self.epsilon:
      observation = observation[np.newaxis, :]
      state = torch.tensor(observation).to(self.q_eval.device)
      _, advantage = self.q_eval.forward(state)
      action = torch.argmax(advantage).item()
    else:
      action = np.random.choice(self.action_space)

    return action

  def replace_target_network(self):
    if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
      self.q_next_eval.load_state_dict(self.q_eval.state_dict())

  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

  def learn(self):
    # What happens if we havent filled up batchsize of our memory
    if self.memory.mem_cntr < self.batch_size:
      return
    
    self.q_eval.zero_grad()

    self.replace_target_network()

    state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

    state = torch.tensor(state).to(self.q_eval.device)
    new_state = torch.tensor(new_state).to(self.q_eval.device)
    action = torch.tensor(action).to(self.q_eval.device)
    reward = torch.tensor(reward).to(self.q_eval.device)
    dones = torch.tensor(done).to(self.q_eval.device)

    V_s, A_s = self.q_eval.forward(state)
    V_s_, A_s_ = self.q_next_eval.forward(new_state)

    q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1, action.unsqueeze(-1)).squeeze(-1)
    q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

    q_target = reward + self.gamma * torch.max(q_next, dim=1)[0].detach()
    q_target[dones] = 0.0

    loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
    loss.backward()
    self.q_eval.optimizer.step()
    self.learn_step_counter += 1

    self.decrement_epsilon()

  def save_models(self):
    self.q_eval.save_checkpoint()
    self.q_next_eval.save_checkpoint()

  def load_models(self):
    self.q_eval.load_checkpoint()
    self.q_next_eval.load_checkpoint()

  

    


    



                                    