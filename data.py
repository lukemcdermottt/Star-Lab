import numpy as np 
import pandas as pd
import torch
from data_augmentation import *
from torch.utils.data import Dataset, DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    def __len__(self):
        return self.data.shape[0]
        
def get_data():
    #Load Dataframes
    bin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/binaries')
    sin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/singles')
     
    #Prep data for NN
    bin_data = bin_df.iloc[:,:441].to_numpy() #np.zeros(bin_df.shape[0])])
    sin_data = sin_df.iloc[:,:441].to_numpy() #np.zeros(bin_df.shape[0])])

    bin_data, sin_data  = reduce_dim(bin_data, sin_data, 256)
    sin_data = generate_data(sin_data, 3, 40000)
    data, labels = add_labels(bin_data, sin_data)
    data, labels = torch.tensor(data), torch.tensor(labels)

    #Reshape into 16x16 image
    data = data.reshape(data.shape[0], 1, 16, 16)
    return split_data(data.float(), labels.float())

def reduce_dim(bin_data, sin_data, dim):
    num_bin = len(bin_data)
    data = np.concatenate((bin_data, sin_data), axis = 0)
    data_p = PCA(data, dim)
    return data_p[:num_bin], data_p[num_bin:]

def generate_data(data, K, amount):
    size = np.shape(data)[0]
    clusters = []
    for _ in range(K):
        clusters.append(np.array([data[0]]))
    
    #Create Clusters
    Kmus, Rnk = runKMeans(K, data, False)
    for count in range(size):
        clusters[np.argmax(Rnk[count])] = np.concatenate((clusters[np.argmax(Rnk[count])], np.array([data[count]])), axis = 0)
    
    #Generate New Data from Cluster Distributions
    gen = np.array([data[0]])
    for c in clusters:
        c = c[1:]
        mean = np.mean(c, axis = 0)
        cov = np.cov(c, rowvar = 0)
        new = np.random.multivariate_normal(mean, cov, int(amount/K))
        gen = np.concatenate((gen, new), axis = 0)

    return gen[1:]

def add_labels(bin_data, sin_data):
    bin_labels = np.full((len(bin_data)),[[0]])
    sin_labels = np.full((len(sin_data)),[[1]])

    data = np.concatenate((bin_data, sin_data), axis = 0)
    norm = np.linalg.norm(data, axis = 0)
    data = data/norm
    labels = np.concatenate((bin_labels, sin_labels), axis = 0)
    
    #Shuffle labels
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]
    return data, labels

def split_data(data, labels):
    #shuffle labels
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    train_data = data[:int(len(data)*0.8)]
    train_labels = labels[:int(len(labels)*0.8)]
    print(train_data.shape[0])
    val_data = data[int(len(data)*0.8):]
    val_labels = labels[int(len(labels)*0.8):]

    return train_data, train_labels, val_data, val_labels

#Will use later
def gen_folds(data, labels, K):
    return data