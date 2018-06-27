import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from model_gen import Net, Netv1
from dataset import ESC50_Dataset
from torchsummary import summary

def save_checkpoint(state, acc):
    torch.save(state, str(acc)+'best.plk')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = Netv1()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

dataset = ESC50_Dataset('esc50.csv', '/home/adam/ESC-50-master/audio/')
dataset_test = ESC50_Dataset(
    'esc50.csv', '/home/adam/ESC-50-master/audio/', folds=[4, 5])

trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=2)

classes = dataset.dictionary
best = 0
prevacc = -1
for epoch in range(1000):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

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
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
    acc = 100 * correct / total
    if acc > prevacc:
        save_checkpoint(net, acc)
        prevacc = acc
print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
