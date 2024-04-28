

latent_dim = 100 
num_clss = 6 
batch_size = 128
import torch
import torch.nn as nn
import torch.nn.functional as F
pretrained = True 

print("STARTED ....")
from gan_model import Generator

from _prep import get_initials, save_pytorch_model, set_seed, visualize

(path_dataset,path_trained_models, 
     path_training_results,path_graphics, device_, 
     num_workers, pretrained,learning_rate, weight_decay, epoch_number) = get_initials()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from EBSDDataset import call_dataloader, call_whole_dataloader, call_partial_dataloader
train_loader, test_loader,  validation_loader, classes = call_dataloader(path = path_dataset, batch_size = batch_size)

from models import get_models, count_parameters
disc_model = get_models("nvidia", pretrained)
disc_model.classifier = nn.Linear(in_features=256, out_features=num_clss+1, bias=True) # 1 for the unsupervised   

import torch.optim as optim

disc_optimizer = optim.Adam(
                        params = disc_model.parameters(),
                        lr = learning_rate,
                        weight_decay = weight_decay
                        )

    
#     # Loss fucntion
disc_sup_criteria = nn.CrossEntropyLoss()
disc_unsup_criteria = nn.BCELoss() #Binary Cross entropy

disc_model.to(device_)

gen_model = Generator(latent_dim, image_size=224, channels=3) #Generator




gen_optimizer = optim.Adam(
                        params = gen_model.parameters(),
                       lr=0.002,
                        betas= (0.5, 0.999)
                        )
gen_model.to(device_)

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
        
# gen_model.apply(weights_init)

def custom_activation(x):
    exp_x = torch.exp(x)
    Z_x = torch.sum(exp_x, dim=-1, keepdim=True)
    D_x = Z_x / (Z_x + 1)
    return D_x

import numpy as np
from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint

def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix].to(device_), labels[ix].to(device_) #Select random images and corresponding labels
    y = torch.ones((n_samples, 1)).to(device_) #Label all images as 1 as these are real images. (for the discriminator training) 
    return [X, labels], y


def generate_latent_points(latent_dim, n_samples):
    # Generate random noise
    # z_input = torch.from_numpy(np.random.uniform(-1, 1, [n_samples, latent_dim]).astype(np.float32))
    # return z_input.to(device_)
    return torch.randn(n_samples, latent_dim).to(device_)


def generate_fake_samples(generator, latent_dim, n_samples):
    
    z_input = generate_latent_points(latent_dim, n_samples)
    fake_images = generator(z_input)
	# create class labels
    y = torch.zeros((n_samples, 1)).to(device_)
	
    return fake_images, y

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def train(gen_model, disc_model, latent_dim, n_epochs=20, n_batch=120):
	
    bat_per_epo = int(len(train_loader.dataset) / n_batch)
	# iterations
    n_steps = bat_per_epo * n_epochs
	
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, 
                                                              n_batch, half_batch, 
                                                              bat_per_epo, n_steps))
    for i in range(n_steps):
		# update supervised discriminator (disc_sup) on real samples.
        #Remember that we use real labels to train as this is supervised. 
        #This is the discriminator we really care about at the end.
        #Also, this is a multiclass classifier, not binary. Therefore, our y values 
        #will be the real class labels for MNIST. (NOT 1 or 0 indicating real or fake.)
        X_sup, y_sup = next(iter(train_loader))

        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
         
        #DISCRIMINATOR 
        
        # disc_model.train()
        disc_optimizer.zero_grad()
        

        y_pred_sup = disc_model(Xsup_real)[0][:,:len(classes)]
        xloss = disc_sup_criteria(y_pred_sup, ysup_real)     
        xloss.backward()
        
        # train on real. 
        [X_real, _], y_real = generate_real_samples([X_sup, y_sup], half_batch)
        y_pred_unsup = disc_model(X_real)[0][:,len(classes):]
        y_pred_unsup = custom_activation(y_pred_unsup)
        yloss = disc_unsup_criteria(y_pred_unsup, y_real)
        yloss.backward()
        
        
        #Now train on fake. 
        X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, half_batch)
        y_pred_unsup = disc_model(X_fake.detach())[0][:,len(classes):]
        y_pred_unsup = custom_activation(y_pred_unsup)
        zloss = disc_unsup_criteria(y_pred_unsup, y_fake)
        zloss.backward()
        
        disc_optimizer.step()    
        
        
        #GENERATOR 
        #gen_model.train()
        gen_optimizer.zero_grad()
        
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), torch.ones((n_batch//2, 1)).to(device_)
        #X_predicted = gen_model(X_gan)

        y_predicted = disc_model(X_fake)[0][:,len(classes):]        
        y_predicted = custom_activation(y_predicted)
        qloss = disc_unsup_criteria(y_gan, y_predicted)
        qloss.backward()
        
        gen_optimizer.step()
    
        _, prediction = torch.max(y_pred_sup, dim=1)
        correct_tensor = prediction.eq(ysup_real.data.view_as(prediction))
        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
        
        
		# summarize loss on this batch
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, xloss, accuracy.item()*100, yloss, zloss, qloss))



train(gen_model, disc_model, latent_dim, n_epochs=epoch_number, n_batch= batch_size)

save_pytorch_model(gen_model, "gen_model", path_trained_models)
save_pytorch_model(disc_model, "disc_model", path_trained_models)

def get_test_accuracy(model, model_name, test_loader, device, criteria, classes):
        
        print("TEST STARTED .... ")
        #Test Values
        test_loss = 0.0
        test_acc = 0.0

        y_pred = []
        y_true = []
        #Model in Evaluatıon mode, no changes in models parameters        
        model.eval()
        
        with torch.no_grad():
            for idx2, (imgs2,clss2) in enumerate(test_loader):
                
                imgs2 = imgs2.to(device)
                imgs2 = imgs2.to(torch.float32)

                if model_name == "nvidia" or model_name == "google_hugging":
                    y_pred2 = model(imgs2)[0][:,:len(classes)]
                    target2 = clss2.to(device)
                
                else:
                    y_pred2 = model(imgs2)
                    target2 = clss2.to(device)
                
                loss2 = criteria(y_pred2, target2)
              
                #On each batch it sum up.
                test_loss += loss2.item()* imgs2.size(0)
                
                _, prediction2 = torch.max(y_pred2, dim=1)
                correct_tensor2 = prediction2.eq(target2.data.view_as(prediction2))
                accuracy2 = torch.mean(correct_tensor2.type(torch.FloatTensor))
                
                y_pred.append(prediction2)
                y_true.append(clss2)
                
                test_acc+= accuracy2.item() * imgs2.size(0)

                
        #Epoch losses and accuracy
        test_loss = test_loss / (len(test_loader.sampler))
        test_acc = test_acc / (len(test_loader.sampler))

        print('Test Loss: %f ' % (test_loss))
        print('Test Acc: %f ' % (test_acc))

get_test_accuracy(disc_model, "nvidia", test_loader,  device_, disc_sup_criteria, classes)

