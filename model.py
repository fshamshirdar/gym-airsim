
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Actor, self).__init__()
        """ W = (W-F+2P) / S+1 """
        self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=(4, 4), stride=4) # 1x30x100 -> 32x7x25     
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2) # 32x7x25 -> 64x3x12
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=1) # 64x3x12 -> 64x3x12
        self.fc1 = nn.Linear(64 * 3 * 12, 512)
        self.fc2 = nn.Linear(512, nb_actions)

        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
	x = F.tanh(self.fc2(x))
	return x

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, init_w=3e-3):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(nb_states, 32, kernel_size=(4, 4), stride=4) # 1x30x100 -> 32x7x25     
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2) # 32x7x25 -> 64x3x12
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=1) # 64x3x12 -> 64x3x12
        self.fcs1 = nn.Linear(64 * 3 * 12, 128)
        self.fca1 = nn.Linear(nb_actions, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
	self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # debug()
        x = F.relu(self.fcs1(x))
        xa = F.relu(self.fca1(a))
        x = F.relu(self.fc2(torch.cat([x,xa],dim=1)))
        x = self.fc3(x)

        return x
