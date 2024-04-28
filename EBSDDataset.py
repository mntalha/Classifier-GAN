#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 23:02:16 2023

@author: talha
"""

from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from scipy import ndimage as nd
from PIL import Image
import cv2

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import sobel 

from _prep  import path_dataset
## ??? Question . Why np.var is calculated during the loss calculation 

class EBSDDataset(Dataset):
    def __init__(self, data_dir, image_size=(224, 224), mode="train"):
        
        self.data_dir = os.path.join(data_dir, mode)  
        self.image_size = image_size
        self.clss = sorted([clss for clss in os.listdir(self.data_dir) if not clss.startswith('.')])
        self.dataset = ImageFolder(self.data_dir)
        self.mode = mode

        # self.train_dir = os.path.join(data_dir, 'train')
        # self.test_dir = os.path.join(data_dir, 'test')
        # self.validation_dir = os.path.join(data_dir, 'val')
        # if mode == "train":
        #     self.build_train_data()
        # elif mode == "val":
        #     self.build_validation_data()
        # elif mode == "test":
        #     self.build_test_data()
            
        # self.transform_operation = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(self.image_size),
        #     ])
        
        self.transform_operation = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([224,224]),
            transforms.ToTensor()
            ])
        
    # def load_data(self, directory):
        
    #     self.dir = directory
    #     #self.images = [] #np.zeros(len_)
    #     #self.classes = [] #np.zeros(len_)
    #     #self.index_csv = [] #np.zeros(len_)
    #     self.imgs_name = [file_name for file_name in sorted(os.listdir(directory)) if file_name.endswith('.png')]#[:10]
    #     self.unique_classes = sorted(set(self.csv[1])) #[198, 216, 221, 225, 227] #self.unique_classes.index()
        # for idx,imgs in enumerate(imgs_name):
        #     #print(idx, imgs)
        #     image_path = os.path.join(directory,imgs) #./ML_data/train/mp-1001016_cubic_227.hspy_227_114807.png
        #     self.images.append(plt.imread(image_path))
        #     self.classes.append(int(image_path.split("_")[-2]))
        #     self.index_csv.append(int((image_path.split("_")[-1]).split(".")[0]))         
    
    # def read_csv_file(self, filePath):
    #     try:
    #         with open(filePath, 'r') as csv_file:
    #             df = pd.read_csv(csv_file, header=None)
    #             return df
    #     except FileNotFoundError:
    #         print(f"The file '{filePath}' could not be found.")
    #     except IOError:
    #         print(f"An error occurred while reading the file '{filePath}'.")       
            
    # def build_train_data(self):
    #     self.csv = self.read_csv_file(train_csv_path)
    #     self.load_data(self.train_dir)
    # def build_test_data(self):
    #     self.csv = self.read_csv_file(test_csv_path)
    #     self.load_data(self.test_dir) 
    # def build_validation_data(self):
    #     self.csv = self.read_csv_file(validation_csv_path)
    #     self.load_data(self.validation_dir)

    def __len__(self):
        return len(self.dataset)  
    
    def __getitem__(self,idx):
        
        ####
        gray_image = self.dataset[idx][0].convert("L")
        image_array = np.array(gray_image)

        ## img = image_array # (nd.gaussian_filter(image_array,sigma = 3))
        std_dev = 25

        # Generate random Gaussian noise
        noise = np.random.normal(0, std_dev, image_array.shape)
        # dft = cv2.dft(np.float32(image_array), flags=cv2.DFT_COMPLEX_OUTPUT)
        # dft_shift = np.fft.fftshift(dft)
        # magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        edges = cv2.Canny(image_array,50,150,apertureSize = 7)
        minLineLength = 255
        maxLineGap = 0.1
        lines = cv2.HoughLinesP(edges,1,np.pi/180,1,minLineLength,maxLineGap)
        for z in lines:
            x1,y1,x2,y2 = z[0]
            cv2.line(image_array,(x1,y1),(x2,y2),(0,255,0),1)
        # #image_array = (nd.gaussian_filter(image_array,sigma = 1))
        if self.mode == "train":
           
                # Add noise to the image
            #noisy_image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)

            img = np.array(gray_image) + noise   + image_array
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            imgs, clss = self.transform_operation(img),  self.dataset[idx][1]
        else: 
            img = np.array(gray_image) + noise   + image_array
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            imgs, clss = self.transform_operation(img),  self.dataset[idx][1]
        ####
        imgs, clss = self.transform_operation(self.dataset[idx][0]),  self.dataset[idx][1]
        
        return imgs, clss

    # def __getitem__(self,idx):
        
    #     image_path = os.path.join(self.dir,self.imgs_name[idx])
    #     image = plt.imread(image_path)
    #     classes = int(image_path.split("_")[-2])
    #     index_csv = int((image_path.split("_")[-1]).split(".")[0])
    #     csv = self.csv.loc[index_csv][3:].to_numpy().astype(np.float32) #0 index 1:name 2:class_number
        
    #     return image, classes,index_csv,csv #torch.from_numpy(csv.to_numpy()[2:].astype(np.float32))

# train_dataset = EBSDDataset(PATH, mode="train")
# test_dataset = EBSDDataset(PATH, mode="test")
# validation_dataset = EBSDDataset(PATH, mode="val")

# trainloader = DataLoader(dataset = validation_dataset, batch_size = 32, shuffle = True)
# for i in trainloader:
#     print(i[2])
def call_partial_dataloader(path, batch_size = 4, shuffle = False):
    
    #create dataset object
    train_dataset = EBSDDataset(data_dir = path, mode="train")
    size = len(train_dataset)
    
    train_size = int(0.85 * size)
    val_size = int(0.15 * size) 
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, size-train_size])

    test_dataset = EBSDDataset(data_dir = path, mode="test")

    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader,  validation_loader, test_dataset.clss

def call_dataloader(path, batch_size = 4, shuffle = False):
    
    #create dataset object
    train_dataset = EBSDDataset(data_dir = path, mode="train")
    test_dataset = EBSDDataset(data_dir = path, mode="test")
    validation_dataset = EBSDDataset(data_dir = path, mode="val")
    
    clss = train_dataset.clss
    train_size = int(0.1 * len(train_dataset))
    val_size = int(0.1 * len(validation_dataset))
    test_size = int(1 * len(test_dataset))

    train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset)- train_size])
    validation_dataset,  _ = random_split(validation_dataset, [val_size, len(validation_dataset)- val_size])
    test_dataset,  _ = random_split(test_dataset, [test_size, len(test_dataset)- test_size])

    trainloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = batch_size, shuffle = False)

    return trainloader, test_loader, validation_loader, clss

def call_whole_dataloader(path, batch_size = 4):

    dataset = EBSDDataset(data_dir = path, mode="train")

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader,  val_loader, dataset.clss


if __name__ == "__main__":
    
    tr, test, val, clss = call_dataloader(path_dataset, batch_size = 32,shuffle=True)
 
    for i in tr:
        print("ss")