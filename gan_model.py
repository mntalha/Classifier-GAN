import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST



import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, image_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size

        self.conv1 = nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU(True)
        
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(True)
        
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(True)
        
        self.conv4 = nn.ConvTranspose2d(256, 128, 2, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(True)
        
        self.conv5 = nn.ConvTranspose2d(128, 64, 2, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(True)
        
        self.conv6 = nn.ConvTranspose2d(64, 32, 2, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU(True)
        
        self.conv7 = nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        input = input.view(input.size(0), self.latent_dim, 1, 1)
        
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        x = self.conv7(x)
        x = self.tanh(x)
        
        return x


