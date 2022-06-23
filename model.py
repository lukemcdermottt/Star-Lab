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
        #define CNN structure here
        self.conv1 = nn.Conv1d(2, 1, 2)
        self.fc1 = nn.Linear(441,128)
        self.fc2 = nn.Linear(128, 39)

    #Forward Propagate through NN
    def forward(self,x):
        x = F.max_pool1d(F.relu(self.conv1(x)), (2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    
def train_model(model, optimizer, scheduler, criterion, train_data, val_data, args):
    #getting some hyperparameters
    batch_size = args['bz']
    epochs = args['epoch']
    
    #allocating the model to the gpu
    device = torch.device("cuda:{}".format(args['device_id']))
    model.to(device) 

    #inititalizing accuracies and losses lists for train and val
    train_acc, train_loss, val_acc, val_loss = [], [], [], []
    
    for epoch in range(epochs):
        print('epoch', epoch + 1)
        epoch_acc,epoch_loss = [], []   #save acc/loss across the epoch

        #for each piece of training data
        for train_pattern in train_data:
            inputs, labels = train_pattern
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()   #Zeroes the weight gradients
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_acc.append(np.mean(np.argmax(outputs.data.cpu().numpy(), axis=1) == labels.data.cpu().numpy()))
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
        for val_pattern in val_data:
            val_inputs, val_labels = val_pattern
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            
            outputs = model.forward(val_inputs)
            loss = criterion(outputs, val_labels)

            epoch_loss.append(loss.item())
            epoch_acc.append(np.mean(np.argmax(outputs.data.cpu().numpy(), axis=1) == val_labels.data.cpu().numpy()))

        #record loss & acc of validation
        val_acc.append(np.mean(epoch_acc)) 
        val_loss.append(np.mean(np.array(epoch_loss)))

        print('Curr Val Loss', val_loss[-1])
        print('Curr Val Acc', val_acc[-1])

        # early stopping if the val loss goes up for two consecutive epochs
        if epoch > 5 and (val_loss[-1] > val_loss[-2]) and (val_acc[-1] > val_acc[-3]):
            print("Early stopping at epoch: ",epoch)
            del val_inputs, val_labels
            torch.cuda.empty_cache()
            break
        
        del val_inputs, val_labels
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