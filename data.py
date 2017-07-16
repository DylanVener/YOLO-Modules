import torch
import os
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import json

class ImdbDataset(Dataset):

    def __init__(self, json_file, root_dir):
        self.root_dir = root_dir
        self.pairs = json.load(open(json_file))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        X_path = os.path.join(self.root_dir, self.pairs[idx][0])
        X = torch.load(X_path)
        y = torch.FloatTensor([self.pairs[idx][1]])

        return {'X' : X, 'y' : y}

    

