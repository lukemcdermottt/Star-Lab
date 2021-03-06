import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Define the Neural Network
class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d((self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return self.sm(x)


"""
def train_model(model, optimizer, scheduler, criterion, train_data, val_data):
    #getting some hyperparameters
    batch_size = 2
    epochs = 100
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    #inititalizing accuracies and losses lists for train and val
    train_acc, train_loss, val_acc, val_loss = [], [], [], []
    
    for epoch in range(epochs):
        print('epoch', epoch + 1)
        epoch_acc,epoch_loss = [], []   #save acc/loss across the epoch

        #for each piece of training data
        for i in range(0, len(train_images)-batch_size, batch_size):
            optimizer.zero_grad()   #Zeroes the weight gradients

            inputs = train_images[i:i+batch_size]
            labels = train_labels[i:i+batch_size]
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels).float()
            
            inputs = inputs.unsqueeze(1)
            print(inputs.shape, labels.shape)
            
            outputs = model.forward(inputs).float()
            #print('outputs shape:', outputs.shape, outputs.dtype)
            #print('labels shape:', labels.shape, labels.dtype)
            #print(outputs[:5], labels[:5])

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            epoch_acc.append(np.mean(((outputs.data.cpu().numpy() - labels.data.cpu().numpy())**2 < .25)))
            epoch_loss.append(loss.item()) 

        #Save Training Loss & Accuracy after 1 run through
        train_acc.append(np.mean(epoch_acc))
        train_loss.append(np.mean(np.array(epoch_loss)))

        print('Curr Train Loss', train_loss[-1])
        print('Curr Train Acc', train_acc[-1])

        #Test against validation
        epoch_acc = []
        epoch_loss = [] 

        for i in range(0, len(val_images)-batch_size, batch_size):
            #get the data
            inputs = val_images[i:i+batch_size]
            labels = val_labels[i:i+batch_size]
            #convert to tensor
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels).float()
        
            inputs = inputs.unsqueeze(1)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            epoch_loss.append(loss.item())
            epoch_acc.append(np.mean(((outputs.data.cpu().numpy() - labels.data.cpu().numpy())**2 < .25)))

        #record loss & acc of validation
        val_acc.append(np.mean(epoch_acc)) 
        val_loss.append(np.mean(np.array(epoch_loss)))

        print('Curr Val Loss', val_loss[-1])
        print('Curr Val Acc', val_acc[-1])
        


    #Plot Loss & Accuracy of Train/Val over each epochs
    plt.figure()
    plt.title('accuracies')
    plt.plot(val_acc, label=f'validation accuracy')
    plt.plot(train_acc, label=f'training accuracy')
    plt.legend()
    plt.show()
    #plt.savefig("./resnet_plots_bz=32_lr=1e-5/train-accuracy.png")
    #plt.close()


    #plt.savefig("resnet_plots_bz=32_lr=1e-5/train-loss.png")

    print()
    print('Train_Acc:', train_acc)
    print()
    print('Val_Acc:', val_acc)
    print('Final losses:', train_loss, val_loss)

    return model # return the model with weight selected by best performance
"""