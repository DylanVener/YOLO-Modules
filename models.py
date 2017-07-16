import torch
import torch.nn as nn
import torch.nn.functional as F

def Sex(nn.Module):
    def __init__(self):
        self.relu = nn.LeakyReLU(0.1)
        
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1000)
        self.fc3 = nn.Linear(1000,2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
