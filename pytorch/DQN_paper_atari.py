import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import retro
from collections import deque

env_name = 'SpaceInvaders-Atari2600'
env = retro.make(game=env_name)
state = env.reset()

print("Observation space size: ", env.observation_space.shape)
print("Action space size: ", env.action_space.n)

state_dim = env.observation_space.shape
action_size = env.action_space.n

# State as a torch tensor
state_tensor = torch.from_numpy(state)
print(state_dim[2])

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
action_size = env.action_space.n # 8 actions
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
    def __init(self, state_dim, action_size, learning_rate):
        super().__init__()
        print("INSTANTIATED DQNetwork.........")
        self.state_dim = state_dim
        self.action_size = action_size
        self.action_in = 0
        self.action_one_hot = F.one_hot(torch.arange(0,action_size)[self.action_in], num_classes=action_size).float()

        # - passes through 3 convnets
        # - flattened
        # - passes through 2 FC layers
        # - Outputs a Q value for each actions

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        def forward(self, x):
            # 3 Convs
            x = F.elu(self.conv1(x))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            # Flatten
            x = x.view(x.shape[0], -1)
            inp_size = x.shape
            print(inp_size, *inp_size)
            # 2 FC's
            self.fc1 = nn.Linear(*inp_size, 100) # Fully connected
            self.q_state = nn.Linear(100, self.action_size)
            x = F.relu(self.fc1(x))
            x = self.q_state(x)
            return x

# Experience Replay
"""
Here we create the Memory object that creates a deque.
"""
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        return [self.buffer[i] for i in index]