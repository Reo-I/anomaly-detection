import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.nn.functional as F
import os
import pylab
import matplotlib.pyplot as plt
import pickle
import numpy as np
import fasttext

from vae import VAE
from visualization import v_loss, v_latent

#-------------------------#
#     main parameters     #
#-------------------------#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 200  #number of epochs
batch_size = 16  #batch size
learning_rate = 1e-5 
train =True
latent_dim = 2

#-------------------------#
#    learning word2vec    #
#-------------------------#
w2v_model = fasttext.train_unsupervised("train.txt", "skipgram", dim=64, minCount=1)

#------------------------------------#
#      learing features of User2     #
#------------------------------------#
with open("masquerade-data/User2") as f:
    cmds=[l.rstrip() for l in f.readlines()]

x=[w2v_model[cmd] for cmd in cmds]
x_train=np.reshape(np.array(x[:4000]),(40,1, 100,64))
x_valid=np.reshape(np.array(x[4000:5000]),(10,1, 100,64))
x_test=np.reshape(np.array(x[5000:]),(-1,1, 100,64))


train_dataloader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(x_valid, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(x_test, batch_size=1, shuffle=False)

#-------------#
#  train vae  #
#-------------#
train_losses = []
val_losses = []


model = VAE(latent_dim = latent_dim, device = device).to(device)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    itr = 0
    train_loss = 0
    val_loss = 0 
    if epoch%5==0:
        print(epoch, "/", num_epochs)
    model.train()
    for data in train_dataloader:  
        model.train()  
        itr+=1
        img = data
        img = Variable(img).to(device)

        if train == False:
            output,mu,var,latent = model(img)            
            
        else:
            #calcurate the loss
            loss,output,mu,var,latent = model.loss(img) 
            train_loss+=loss.detach().cpu().numpy()
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()                
            #update parameters
            optimizer.step()                
            #print('{} {}'.format(itr,loss))
    train_losses.append(train_loss/40.)
    model.eval()
    for data in val_dataloader:
        img = data
        img = Variable(img).to(device)
        loss,output,mu,var,latent = model.loss(img) 
        val_loss+=loss.detach().cpu().numpy()
    val_losses.append(val_loss/10.)

#-------------------------------------------------#
#  validate trained vae for train/val/test data   #
#-------------------------------------------------#
train_z = np.array([[0, 0]])
val_z = np.array([[0, 0]])
test_z = np.array([[0, 0]])
test_each_loss = []
model.eval()
for data in train_dataloader:
    img = Variable(data).to(device)
    loss, output,mu,var,latent = model.loss(img)
    train_z = np.append(train_z, latent.cpu().detach().numpy(), axis = 0)
for data in val_dataloader:
    img = Variable(data).to(device)
    loss, output,mu,var,latent = model.loss(img)
    val_z = np.append(val_z, latent.cpu().detach().numpy(), axis = 0)
    #val_each_loss.append(loss.detach().cpu().numpy())
for data in test_dataloader:
    img = Variable(data).to(device)
    loss, output,mu,var,latent = model.loss(img)
    test_z = np.append(test_z, latent.cpu().detach().numpy(), axis = 0)
    test_each_loss.append(loss.detach().cpu().numpy())
#z = latent.cpu().detach().numpy()
#num = num.cpu().detach().numpy()

if latent_dim == 2:
    plt.figure(figsize=(15, 15))
    plt.scatter(train_z[1:, 0], train_z[1:, 1], marker='.',  c = "black")
    plt.scatter(val_z[1:, 0], val_z[1:, 1], marker='.',  c = "blue")
    plt.scatter(test_z[1:, 0], test_z[1:, 1], marker='.',  c = "red")
    plt.grid()
    plt.savefig("./fig.png")
if train == True:
    torch.save(model.state_dict(), './conv_Variational_autoencoder_{}dim.pth'.format(latent_dim))

v_loss(train_losses, val_losses)