import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, n_classes = 50):
        super(Net, self).__init__()
        # network setup
        self.conv1 = nn.Conv2d(1, 32, (1, 64), stride = (1,2),bias= False)
        self.bn1 = nn.BatchNorm2d(32)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(32, 64, (1, 16), stride = (1,2),bias= False)
        self.bn2 = nn.BatchNorm2d(64)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(1, 32, (8, 8),bias= False)
        self.bn3 = nn.BatchNorm2d(32)
        self.elu3 = nn.ELU()
        self.conv4 = nn.Conv2d(32, 32, (8, 8),bias= False)
        self.bn4 = nn.BatchNorm2d(32)
        self.elu4 = nn.ELU()
        self.conv5 = nn.Conv2d(32, 64, (1, 4),bias= False)
        self.bn5 = nn.BatchNorm2d(64)
        self.elu5 = nn.ELU()
        self.conv6 = nn.Conv2d(64, 64, (1, 4),bias= False)
        self.bn6 = nn.BatchNorm2d(64)
        self.elu6 = nn.ELU()
        self.conv7 = nn.Conv2d(64, 128, (1, 2),bias= False)
        self.bn7 = nn.BatchNorm2d(128)
        self.elu7 = nn.ELU()
        self.conv8 = nn.Conv2d(128, 128, (1, 2),bias= False)
        self.bn8 = nn.BatchNorm2d(128)
        self.elu8 = nn.ELU()
        self.conv9 = nn.Conv2d(128, 256, (1, 2),bias= False)
        self.bn9 = nn.BatchNorm2d(256)
        self.elu9 = nn.ELU()
        self.conv10 = nn.Conv2d(256, 256, (1, 2),bias= False)
        self.bn10 = nn.BatchNorm2d(256)
        self.elu10 = nn.ELU()
        self.fc11 = nn.Linear(256 * 10 * 8, 4096)
        self.fc12 = nn.Linear(4096, 4096)
        self.fc13 = nn.Linear(4096, 50)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.elu1(self.bn1(self.conv1(x)))
        x = self.elu2(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (1, 64))
        x = x.transpose(1, 2)
        x = self.elu3(self.bn3(self.conv3(x)))
        x = self.elu4(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, (5, 3))
        x = self.elu5(self.bn5(self.conv5(x)))
        x = self.elu6(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, (1, 2))
        x = self.elu7(self.bn7(self.conv7(x)))
        x = self.elu8(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, (1, 2))
        x = self.elu9(self.bn9(self.conv9(x)))
        x = self.elu10(self.bn10(self.conv10(x)))
        x = F.max_pool2d(x, (1, 2))
        x = F.elu(self.fc11(x))
        x = F.dropout(x, training=True)
        x = F.elu(self.fc12(x))
        x = F.dropout(x, training=True)
        return nn.Softmax(self.fc13(x))

class Netv1(nn.Module):

    def __init__(self, n_classes = 50):
        super(Netv1, self).__init__()
        # network setup
        self.conv1 = nn.Conv2d(1, 40, (1, 8),bias= False, padding=0)
        self.bn1 = nn.BatchNorm2d(40)
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv2d(40, 40, (1, 8),bias= False)
        self.bn2 = nn.BatchNorm2d(40)
        self.elu2 = nn.ELU()
        self.conv3 = nn.Conv2d(1, 50, (8, 13),bias= False)
        self.bn3 = nn.BatchNorm2d(50)
        self.elu3 = nn.ELU()
        self.conv4 = nn.Conv2d(50, 50, (1, 5),bias= False)
        self.bn4 = nn.BatchNorm2d(50)
        self.elu4 = nn.ELU()
        self.fc5 = nn.Linear(50*11 *1, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 50)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.elu1(self.bn1(x))
        
        x = self.elu2(self.bn2(self.conv2(x)))
        
        x = F.max_pool2d(x, (1, 160))
        x = x.transpose(1, 2)
        
        x = self.elu3(self.bn3(self.conv3(x)))
        
        x = F.max_pool2d(x, (3,3))
        x = self.elu4(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, (1, 3))
        x = F.elu(self.fc5(x))
        x = F.dropout(x, training=True)
        x = F.elu(self.fc6(x))
        x = F.dropout(x, training=True)
        return nn.Softmax(self.fc7(x))

