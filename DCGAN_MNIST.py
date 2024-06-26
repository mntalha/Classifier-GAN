#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import matplotlib.pyplot as plt


# In[2]:


# CUDA = False
#DATA_PATH = './data'
# BATCH_SIZE = 128
# IMAGE_CHANNEL = 1
# Z_DIM = 100
# G_HIDDEN = 64
# X_DIM = 64
# D_HIDDEN = 64
# EPOCH_NUM = 5
# REAL_LABEL = 1
# FAKE_LABEL = 0
# lr = 2e-4
# seed = 1


# In[3]:


latent_dim = 100 
num_clss = 6 
batch_size = 16
import torch
import torch.nn as nn
import torch.nn.functional as F
pretrained = True 
from _prep import get_initials, save_pytorch_model, set_seed, visualize
(path_dataset,path_trained_models, 
     path_training_results,path_graphics, device_, 
     num_workers, pretrained,learning_rate, weight_decay, epoch_number) = get_initials()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from EBSDDataset import call_dataloader, call_whole_dataloader, call_partial_dataloader
train_loader, test_loader,  validation_loader, classes = call_dataloader(path = path_dataset, batch_size = batch_size)


# In[4]:


CUDA = True
IMAGE_CHANNEL = 3
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 64
D_HIDDEN = 64
EPOCH_NUM = 25
REAL_LABEL = 1
FAKE_LABEL = 0
seed = 1
BATCH_SIZE = batch_size
lr = 2e-4


# In[5]:


CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if CUDA:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if CUDA else "cpu")
cudnn.benchmark = True


# In[6]:


#ata preprocessing
# dataset = dset.MNIST(root=DATA_PATH, download=True,
#                      transform=transforms.Compose([
#                      transforms.Resize(X_DIM),
#                      transforms.ToTensor(),
#                      transforms.Normalize((0.5,), (0.5,))
#                      ]))

# # Dataloader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
#                                          shuffle=True, num_workers=2)


# In[7]:


next(iter(train_loader))[0].shape


# In[8]:


# Plot training images
real_batch = next(iter(train_loader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# In[9]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[38]:


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # input layer
#             nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(G_HIDDEN * 8),
#             nn.ReLU(True),
#             # 1st hidden layer
#             nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 2, bias=False),
#             nn.BatchNorm2d(G_HIDDEN * 4),
#             nn.ReLU(True),
#             # 2nd hidden layer
#             nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 3, 1, bias=False),
#             nn.BatchNorm2d(G_HIDDEN * 2),
#             nn.ReLU(True),
#             # 3rd hidden layer
#             nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(G_HIDDEN),
#             nn.ReLU(True),
#             # output layer
#             nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 4, 4, bias=False),
#             nn.Tanh()
#         )

#     def forward(self, input):
#         return self.main(input)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input layer
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            # 1st hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            # 2nd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            # 3rd hidden layer
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# In[39]:


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # 1st layer
#             nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 3, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 2nd layer
#             nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(D_HIDDEN * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 3rd layer
#             nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(D_HIDDEN * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 4th layer
#             nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(D_HIDDEN * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # output layer
#             nn.Conv2d(D_HIDDEN * 8, 1, 4, 2, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input).view(-1, 1).squeeze(1)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# In[40]:


# Create the generator
netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

# Create the discriminator
netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)


# In[41]:


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that I will use to visualize the progression of the generator
viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


# In[42]:


dataloader = train_loader


# In[43]:


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader, 0):

        # (1) Update the discriminator with real data
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # (2) Update the discriminator with fake data
        # Generate batch of latent vectors
        noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(FAKE_LABEL)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # (3) Update the generator with fake data
        netG.zero_grad()
        label.fill_(REAL_LABEL)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, EPOCH_NUM, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == EPOCH_NUM-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(viz_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


# In[16]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:





# In[20]:


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.imsave("Original.jpg", np.transpose(vutils.make_grid(real_batch[0].to(device)[:32], padding=2, normalize=True).cpu().numpy(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow()
plt.imsave("Fake.jpg", np.transpose(img_list[-1],(1,2,0)))
plt.show()


# In[19]:


# noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)


# # In[ ]:


# fake = netG(noise)


# # In[ ]:


# plt.subplot(1,2,2)
# plt.axis("off")
# plt.title("Fake Images")
# plt.imshow(np.transpose(fake[-1].detach().numpy(),(1,2,0)))
# plt.show()


# In[ ]:




