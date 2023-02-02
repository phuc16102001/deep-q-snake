import sys
sys.path.insert(0,'.')

import torch.nn as nn
import torch.nn.functional as f

class Linear_QNet(nn.Module):

    def __init__(self, input_size, hidden_size1, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size1)
        self.linear2 = nn.Linear(hidden_size1,output_size)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        return x