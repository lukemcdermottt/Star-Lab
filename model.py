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
        self.conv1 = nn.Conv1d(4,1,2)
        self.fc1 = nn.Linear(1,4)
        self.fc2 = nn.Linear(4,16)
        self.fc3 = nn.Linear(16,1)
        self.sm = nn.Sigmoid()

    #Forward Propagate through NN
    def forward(self,x):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sm(x)
        return x

    
def train_model(model, optimizer, scheduler, criterion, train_data, val_data):
    #getting some hyperparameters
    batch_size = 4
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
            #get the data
            inputs = train_images[i:i+batch_size]
            labels = train_labels[i:i+batch_size]
            #convert to tensor
            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels).float()

            optimizer.zero_grad()   #Zeroes the weight gradients
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

        del inputs, labels
        torch.cuda.empty_cache()

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
        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            epoch_loss.append(loss.item())
            epoch_acc.append(np.mean(((outputs.data.cpu().numpy() - labels.data.cpu().numpy())**2 < .25)))

        #record loss & acc of validation
        val_acc.append(np.mean(epoch_acc)) 
        val_loss.append(np.mean(np.array(epoch_loss)))

        print('Curr Val Loss', val_loss[-1])
        print('Curr Val Acc', val_acc[-1])
        """
        # early stopping if the val loss goes up for two consecutive epochs
        if epoch > 5 and (val_loss[-1] > val_loss[-2]) and (val_acc[-1] > val_acc[-3]):
            print("Early stopping at epoch: ",epoch)
            torch.cuda.empty_cache()
            break
        """
        torch.cuda.empty_cache()


    #Plot Loss & Accuracy of Train/Val over each epochs
    plt.figure()
    plt.title('accuracies')
    plt.plot(val_acc, label=f'validation accuracy')
    plt.plot(train_acc, label=f'training accuracy')
    plt.legend()
    plt.show()
    #plt.savefig("./resnet_plots_bz=32_lr=1e-5/train-accuracy.png")
    #plt.close()

    plt.figure()
    plt.title('loss over epochs')
    plt.plot(val_loss, label=f'validation loss')
    plt.plot(train_loss, label=f'training loss')
    plt.legend()
    plt.show()
    #plt.savefig("resnet_plots_bz=32_lr=1e-5/train-loss.png")

    print()
    print('Train_Acc:', train_acc)
    print()
    print('Val_Acc:', val_acc)
    print('Final losses:', train_loss, val_loss)

    return model # return the model with weight selected by best performance