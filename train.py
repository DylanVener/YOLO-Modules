import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import sys

from models import Sex
from data import ImdbDataset


label_file = sys.argv[1]
root_dir = sys.argv[2]
dataset = ImdbDataset(label_file, root_dir)
loader = DataLoader(dataset, batch_size = 4, shuffle=True, num_workers = 2, pin_memory = True)

model = Sex().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epoch = tqdm(enumerate(loader))

for cnt, data in epoch:
    X, label = data['X'], data['y']
    X, label = Variable(X.cuda()), Variable(label.cuda())

    optimizer.zero_grad()

    output = model(X)
    print(output)
    print(label)
    loss = criterion(output, label)
    optimizer.step()

    if cnt % 1000 == 999:
        epoch.write(loss.data)

