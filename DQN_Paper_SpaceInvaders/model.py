import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("RUNNING ON GPU")
else:
    print("RUNNING ON CPU")

print("CURRENT DEVICE INDEX: ",torch.cuda.current_device())
print("NUMBER OF GPUS AVAILABLE: ", torch.cuda.device_count())
torch.cuda.get_device_capability(device=torch.cuda.current_device())
print("GPU DEVICE NAME: ", torch.cuda.get_device_name(torch.cuda.current_device()))
print("MAX GPU memory occupied by tensors in bytes: ", torch.cuda.max_memory_allocated(torch.cuda.current_device()))

# Preprocess the image to chuck out unwanted information and reduce no. of pixels involved(bottom part of the game and rgb channels to gray as in section 4.1 of the paper)
def preprocess_frame(frame):
    gray = rgb2gray(frame)
    # Remove the part of frame below the player
    cropped_frame = gray[8:-12, 4:-12]
    # Normalize pixel values
    normalized_frame = cropped_frame/255.0
    # Resize
    preprocess_frame = transform.resize(cropped_frame, [110, 84])
    # 110x84x1 frame
    return preprocess_frame 

# skip 4 frames
stack_size = 4
 # Init deque with zero-images one array for each image
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen = 4)

def stacked_frames(stacked_frames, state, is_new_episode):
     # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        #  Clear our stacked_frames
        stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen = 4)

        # Because we're in a new episode, copy the same frame 4 times
        for _ in range(4):
            print("HA")
            stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dim specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames

# MODEL HYPERPARAMETERS
state_size = [110, 84, 4]
# action_size = env.action_space.n # 8 actions
learning_rate = 0.00025

# TRAINING HYPERPARAMS
total_episodes = 50
max_steps = 50000
batch_size = 64

# Exploration params for epsilon greedy strategy
explore_start = 1.0
explore_stop = 0.01   # min exploration probability
decay_rate = 0.00001

# Q Learning hyperparams
gamma = 0.9

# Memory Hyperparams
pretrain_length = batch_size     # No. of experiences stored in the memory when initialized for the first time
memory_size = 1000000       # No. of experiences the memory can keep

# Preprocessing hyperparams
stack_size = 4

# Modify this to false if you just want to see the trained agent
training = False
episode_render = False

# DQN Model
# - Stack of 4 frames as input
# - passes through 3 convnets
# - flattened
# - passes through 2 FC layers
# - Outputs a Q value for each actions

# Space Invaders DQN
class DQNetwork(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()
        print("INSTANTIATING DQNetwork.........")

        self.action_size = 8
        self.action_in = 0
        self.action_one_hot = F.one_hot(torch.arange(0,self.action_size)[self.action_in], num_classes=self.action_size).float()
        self.lr = learning_rate

        # - passes through 3 convnets
        # - flattened
        # - passes through 2 FC layers
        # - Outputs a Q value for each actions

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.fc1 = nn.Linear(128*19*8, 512)
        self.fc2 = nn.Linear(512, self.action_size)

        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        def forward(self, observation):
            observation = torch.Tensor(observation).to(self.device)
            observation = observation.view(-1, 1, 185, 95)
            # 3 Convs
            observation = F.elu(self.conv1(observation))
            observation = F.elu(self.conv2(observation))
            observation = F.elu(self.conv3(observation))
            # Flatten
            observation = observation.view(-1, 128*19*8)
            # 2 FC's
            observation = F.relu(self.fc1(observation))
            """
            Though this is named actions, the output of the NN is actually Q-values 
            for each action which we would then multiply with one of the 8 hot 
            encoded actions to get one Q-Value. actions will not be 1x8 as one would expect.
            It's dimensions would be (no.of images passed to the network) nx8 (actions)
            """
            actions = self.fc2(observation) 
            return observation

class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                 maxMemorySize, epsEnd=0.05,
                 replace=10000, actionSpace=[0,1,2,3,4,5,6,7]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize   # For efficiency, to keep track of subset of state, action, reward triplets
        self.steps = 0  
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.Q_eval = DQNetwork(alpha)
        self.Q_next = DQNetwork(alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        
        if self.learn_step_counter % self.replace_target_cnt == 0:
           self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.memCntr + batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memCntr - batch_size - 1)))

        # miniBatch size can only be batch_size from memStart
        miniBatch = self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        # [:, 0] -> all rows of the batch?, index 0 represents state, 3 is next_state
        # [:] -> all images
        Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        # dim 1 corresponds to actions, 2 to rewards
        maxA = torch.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = torch.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Qtarget = Qpred
        Qtarget[:, maxA] = rewards + self.GAMMA*torch.max(Qnext[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END
        
        loss = self.Q_eval.loss(Qtarget - Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

