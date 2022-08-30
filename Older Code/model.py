from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader, TensorDataset

cudnn.benchmark = True
plt.ion()   # interactive mode

num_samples = 35000

df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/binaries')
df = df.loc[df['primary_type'] <= df['secondary_type']]
pca = PCA(n_components=224)
df = df.sample(num_samples)
X=df.iloc[:,:440].values    #grab flux values
pca.fit(X)
x=np.expand_dims(pca.transform(X), axis = 1)
print(np.shape(x))
labels_df =df.iloc[:,441:443].values #grab labels
labels_os = np.zeros((num_samples, 24*24))

#one hot encode
for idx, i in enumerate(labels_df):
    a = np.zeros(24*24)
    try:
        a[24*(int(i[0]-16)) + int((i[1]-16))] = 1
    except:
        print(i)
    labels_os[idx] = a

print(np.shape(labels_os))
images= torch.tensor(x,dtype=torch.float32)
labels= torch.tensor(labels_os,dtype=torch.float32)
images = nn.functional.normalize(images)

train_images = images[:int(len(images)*0.8)]
train_labels = labels[:int(len(labels)*0.8)]
test_images = images[int(len(images)*0.8):]
test_labels = labels[int(len(labels)*0.8):]

dataloaders = {'train': torch.utils.data.DataLoader(TensorDataset(train_images, train_labels), batch_size=1, shuffle=True, num_workers=4), 
                'val': torch.utils.data.DataLoader(TensorDataset(test_images, test_labels), batch_size=1, shuffle=True, num_workers=4)}
dataset_sizes = {'train': len(train_images),
                'val': len(test_images)}

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 5, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(5, 15, 5)
        self.fc1 = nn.Linear(795, 24*24*3)
        self.fc2 = nn.Linear(24*24*3, 24*24*2)
        self.fc3 = nn.Linear(24*24*2, 24*24)
        self.s = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.s(self.fc3(x))
        return x


net = Net()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print('output', outputs[0], 'label', labels[0])
                    preds = torch.argmax(outputs, 1)
                    #print(outputs[0], labels[0])
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #print(preds, labels.data, labels)
                
                running_corrects += torch.sum(preds == torch.argmax(labels.data,1))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
        print(outputs[0], labels[0])

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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