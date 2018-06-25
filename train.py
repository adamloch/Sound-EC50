import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from model_gen import Net, Netv1
from dataset import ESC50_Dataset
from torchsummary import summary



net = Netv1()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

dataset = ESC50_Dataset('esc50.csv', '/home/adam/ESC-50-master/audio/')
dataset_test = ESC50_Dataset(
    'esc50.csv', '/home/adam/ESC-50-master/audio/', folds=[4, 5])

trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=1)

classes = dataset.dictionary

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
