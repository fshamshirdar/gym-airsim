import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=3):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        """ W = (W-F+2P) / S+1 """
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(4, 4), stride=4) # 1x30x100 -> 32x7x25     
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2) # 32x7x25 -> 64x3x12
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=1) # 64x3x12 -> 64x3x12
        self.fc1 = nn.Linear(64 * 3 * 12, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
