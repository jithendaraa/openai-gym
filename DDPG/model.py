import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DDPGNet(nn.Module):
    def __init__(self):
        super(DDPGNet, self).__init__()
        