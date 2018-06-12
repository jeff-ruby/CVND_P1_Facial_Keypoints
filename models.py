## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net_Basic(nn.Module):

    def __init__(self):
        super(Net_Basic, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.conv1 = nn.Conv2d(1, 32, 5, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 4, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 2, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 384, 1, padding=0)
        self.bn4_1 = nn.BatchNorm2d(384)

        self.fc1 = nn.Linear(13824, 1000)
        self.bn5 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn6 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(1000, 136)
            
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # conv1
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))

        # conv2
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        
        # conv3
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))

        # conv4
        x = self.bn4(self.conv4(x))
        x = self.pool(F.relu(x))
        
        # conv5
        x = self.bn4_1(self.conv5(x))
        x = self.pool(F.relu(x))

        # flatten
        x = x.view(x.size()[0], -1)

        # fc1
        x = self.bn5(self.fc1(x))
        x = F.relu(x)
        
        # fc2
        x = self.bn6(self.fc2(x))
        x = F.relu(x)
        
        # fc3
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.pool = nn.AvgPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 5, padding=0)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 4, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 2, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 384, 1, padding=0)
        self.bn4_1 = nn.BatchNorm2d(384)

        self.fc1 = nn.Linear(13824, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 136)
            
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # conv1
        x = self.bn1(self.conv1(x))
        x = self.pool(F.relu(x))
        # conv2
        x = self.bn2(self.conv2(x))
        x = self.pool(F.relu(x))
        # conv3
        x = self.bn3(self.conv3(x))
        x = self.pool(F.relu(x))
        # conv4
        x = self.bn4(self.conv4(x))
        x = self.pool(F.relu(x))
        # conv5
        x = self.bn4_1(self.conv5(x))
        x = self.pool(F.relu(x))

        # flatten
        x = x.view(x.size()[0], -1)

        # fc1
        x = self.bn5(self.fc1(x))
        x = F.relu(x)
        # fc2
        x = self.bn6(self.fc2(x))
        x = F.relu(x)
        # fc3
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
class Net_VGG_Like(nn.Module):

    def __init__(self):
        super(Net_VGG_Like, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        self.pool = nn.AvgPool2d(2, 2)

        # block 1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=0)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=0)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=0)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=0)
        self.bn2_2 = nn.BatchNorm2d(128)
        
        # block 3
        self.conv3_1 = nn.Conv2d(128, 256, 2, padding=0)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 2, padding=0)
        self.bn3_2 = nn.BatchNorm2d(256)
        
        # block 4
        self.conv4_1 = nn.Conv2d(256, 512, 1, padding=0)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=0)
        self.bn4_2 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(6400, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.drop = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(1000, 136)
            
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # block 1
        x = self.bn1_1(self.conv1_1(x))
        x = self.pool(F.relu(x))
        x = self.bn1_2(self.conv1_2(x))
        x = self.pool(F.relu(x))
        
        # block 2
        x = self.bn2_1(self.conv2_1(x))
        x = self.pool(F.relu(x))
        #x = self.bn2_2(self.conv2_2(x))
        #x = self.pool(F.relu(x))
        
        # block 3
        x = self.bn3_1(self.conv3_1(x))
        x = self.pool(F.relu(x))
        x = self.bn3_2(self.conv3_2(x))
        x = self.pool(F.relu(x))
        
        # block 4
        #x = self.bn4_1(self.conv4_1(x))
        #x = self.pool(F.relu(x))
        #x = self.bn4_2(self.conv4_2(x))
        #x = self.pool(F.relu(x))
        
        # global average pooling
        #x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        
        # flatten
        x = x.view(x.size()[0], -1)

        # fc1
        x = self.bn5(self.fc1(x))
        x = F.relu(x)
        # fc2
        x = self.bn6(self.fc2(x))
        x = F.relu(x)
        # fc3
        #x = self.drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x