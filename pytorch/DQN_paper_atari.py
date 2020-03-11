import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import retro

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



# Space Invaders DQN
class DQNetwork(nn.Module):
    def __init(self, state_dim, action_size):
        super().__init__()
        print("INSTANTIATED DQNetwork.........")

        self.action_in = 0
        self.action_one_hot = F.one_hot(torch.arange(0,action_size)[self.action_in], num_classes=action_size).float()

        # - passes through 3 convnets
        # - flattened
        # - passes through 2 FC layers
        # - Outputs a Q value for each actions

        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= , kernel_size=)
        self.conv2 = nn.Conv2d(in_channels= 3, out_channels= , kernel_size=)
        self.conv3 = nn.Conv2d(in_channels= 3, out_channels= , kernel_size=)


        



class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_space = env.action_space.n
        self.q_network = QNetwork(self.state_sim, self.action_size)
        
        # Discounting rate
        self.gamma = 0.97
        # Epsilon for eps-greedy strategy(explore-exploit dilemma)
        self.eps = 1.0