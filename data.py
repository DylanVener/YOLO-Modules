import torch
import os
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import json

class ImdbDataset(Dataset):

    def __init__(self, json_file, root_dir, avgpool=True):
        self.root_dir = root_dir
        self.pairs = json.load(open(json_file))
        self.avgpool = avgpool

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        X_path = os.path.join(self.root_dir, self.pairs[idx][0])
        X = torch.load(X_path)
        if self.avgpool:
            X = F.max_pool2d(Variable(X), kernel_size=X.size()[2:])
        else:
            X = F.max_pool2d(Variable(X), kernel_size=X.size()[2:])

        X = X.data
        X = X.view(X.size(1)*X.size(2)*X.size(3))

        y = torch.LongTensor([self.pairs[idx][1]])[0]

        return {'X' : X, 'y' : y}



