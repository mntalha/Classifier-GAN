#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 21:24:42 2023

@author: talha
"""

#Libraries 
import torch
import random
import numpy as np
import os


num_workers = 8
epoch_number = 5
learning_rate = 3e-4
weight_decay = 3e-6
model_name = None
pretrained = False 

#paths on server
path = "/data/mnk7465/2-ebsd/"
device_ = "cuda:0" # "cuda:0" "cuda:1" "cpu"

#paths on local
# path = "/Users/talha/Desktop/2-ebsd/"
# device_ = "cpu" # "cuda:0" "cuda:1" "cpu"


path_dataset = path + "dataset/original_data_0" #test_data test_data  concat_data_1  original_data_0 alfred_splitted_data/split0
path_outputs = "/data/mnk7465/2-ebsd/code/Classifier-GAN/" + "outputs/"
path_graphics = path_outputs + "plots/"
path_training_results = path_outputs + "trainining_results"
path_trained_models =  path_outputs + "models"


def get_initials():
    
    return (path_dataset, path_trained_models, path_training_results, path_graphics, 
            device_, num_workers, pretrained, learning_rate, weight_decay, epoch_number)


def set_seed(seed = 42):
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 

def save_pytorch_model(model, model_name, saved_path):
    
    # Check the planned path whether it is exist 
    isExist = os.path.exists(saved_path)
    if not isExist:
        print("Path you wished the model to be saved is not valid...")
        return False
    
    # model.state_dict():
        
    # save it
    path = os.path.join(saved_path, model_name+".pt")   
    torch.save(model.state_dict(), path)
    
def load_pytorch_model(path, raw_model):   
    """
    take the transform .cuda() and .cpu() into consideration.
    """
    import os 
    # Check the file whether it is exist
    isExist = os.path.isfile(path)
    if not isExist:
        print("model couldnt found...")
        return False
    if torch.cuda.is_available():
        raw_model.load_state_dict(torch.load(path))
    else:
        raw_model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

    return raw_model

import matplotlib.pyplot as plt

def visualize(train_loss, validation_loss, title, img_name, epoch_number, status= "Loss"):

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=500)

    plt.title(title)

    color = 'tab:purple'
    plt.plot(train_loss, color=color)

    color = 'tab:blue'
    x_axis = list(range(0, epoch_number, 3))
    plt.plot(x_axis, validation_loss, color=color)

    class_names = ["Train", "Validation"]

    plt.xlabel("Epoch")
    plt.ylabel(status + " Value")

    plt.legend(class_names, loc=1)
    plt.show()

    fig.savefig(img_name, dpi=500)

def save_iterations(path, name):
    
    path = "./outputs/training_results/"+name+".pickle"
    
    with open(path, 'wb') as f:
        pickle.dump(record, f)