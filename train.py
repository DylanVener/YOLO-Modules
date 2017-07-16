import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

from models import Sex
from data import ImdbDataset

dataset = ImdbDataset('./current_ftg.json','')
loader = DataLoader(dataset, batch_size = 4, shuffle=True, num_workers = 2, pin_memory = True)

model = Sex().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epoch = tqdm(enumerate(loader))

for cnt, data in epoch:
    X, label = data['X'], data['label']
    X, label = Variable(X.cuda()), Variable(label.cuda())

    optimizer.zero_grad()

    output = model(X)
    loss = criterion(output, label)
    optimizer.step()
    
    if cnt % 1000 == 999:
        epoch.write(loss.data)

