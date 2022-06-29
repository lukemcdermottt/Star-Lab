import numpy as np 
import pandas as pd
from data_augmentation import *

def get_data():
    #Load Dataframes
    bin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/binaries')
    sin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/singles')
     
    #Prep data for NN
    bin_data = bin_df.iloc[:,:441].to_numpy() #np.zeros(bin_df.shape[0])])
    sin_data = sin_df.iloc[:,:441].to_numpy() #np.zeros(bin_df.shape[0])])

    return bin_data, sin_data

def reduce_dim(bin_data, sin_data):
    num_bin = len(bin_data)
    data = np.concatenate((bin_data, sin_data), axis = 0)
    data_p = PCA(data, 2)
    return data_p[:num_bin], data_p[num_bin:]

def generate_data(data, K, amount):
    size = np.shape(data)[0]
    clusters = 
    gen = np.empty(0)
    #Create Clusters
    Kmus, Rnk = runKMeans(K, data)
    for count in range(size):
        print(type(data[count]))
        clusters[np.argmax(Rnk[count])] = clusters[np.argmax(Rnk[count])].append(data[count])

    print(clusters[0])

    for c in clusters:
        c = np.array(c)
        mean = np.mean(c, axis = 0)
        cov = np.cov(c, rowvar = 0)
        new = np.random.multivariate_normal(mean, cov, int(amount/K))
        gen = np.concatenate((gen, new), axis = 0)

    print('generated!')
    return gen

def add_labels(bin_data, sin_data):
    bin_labels = np.full((len(bin_data)),[[0]])
    sin_labels = np.full((len(sin_data)),[[1]])

    data = np.concatenate((bin_data, sin_data), axis = 0)
    labels = np.concatenate((bin_labels, sin_labels), axis = 0)
    
    #shuffle data
    shuffler = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffler)
    shuffled_data = data[shuffler], labels[shuffler]

    return shuffled_data

#Lets not use this yet
def gen_folds(data, labels, K):
    #Split data into K folds
    n = np.shape(data)[0]
    n_per_fold = int(n/K)
    folds = []
    for i in range(K):
        idx_start = i*n_per_fold
        idx_end = (i+1)*n_per_fold
        folds.append((data[idx_start : idx_end], labels[idx_start : idx_end]))
    return folds



def get_old():
    #Load Dataframes
    bin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/binaries')
    sin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/singles')
    #wav_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/wavegrid')

    #Prep data for NN
    bin_data = bin_df.iloc[:,:441].to_numpy() #np.zeros(bin_df.shape[0])])
    sin_data = sin_df.iloc[:,:441].to_numpy() #np.zeros(bin_df.shape[0])])

    data = bin_data
    labels = np.array([[0]])
    sin_labels = np.array([[1]])
    for i in range(1, len(bin_data)):
        labels = np.concatenate((labels, np.array([[0]])), axis = 0)
    for i in range(1, len(sin_data)):
        sin_labels = np.concatenate((sin_labels, np.array([[1]])), axis = 0)
    for _ in range (90):
        data = np.concatenate((data, sin_data), axis = 0)
        labels = np.concatenate((labels, sin_labels), axis = 0)
    
    #shuffle data
    shuffler = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffler)
    shuffled_data = data[shuffler], labels[shuffler]

    n_zeros = np.count_nonzero(labels==[0])
    n_ones = np.count_nonzero(labels==[1])
    print(n_zeros, n_ones, n_zeros + n_ones)
    return shuffled_data


#PCA'ED DATA
def get_PCA_Single():
    bin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/binaries')
    sin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/singles')
    
    df = sin_df.iloc[:,:441]
    num_singles = len(df)
    df = df.append(bin_df.iloc[:,:441])
    df_p = pd.DataFrame(PCA(df.to_numpy()))
    
    singles = df_p.iloc[:num_singles,:]
    singles.insert(2, 'spectral_type', sin_df['spectral_type'], True)
    sin_images = singles.iloc[:,:2].to_numpy()  #use in nn
    sin_labels = singles.iloc[:,2].to_numpy()   #use in nn
    #Add dimeension to sin_labels
    temp = []
    for i in range(len(sin_images)):
        temp.append([sin_labels[i]])
    sin_labels = np.array(temp)

    shuffler = np.arange(np.shape(sin_images)[0])
    np.random.shuffle(shuffler)
    shuffled_images = sin_images[shuffler], sin_labels[shuffler]
    
    return shuffled_images

def get_m():
    sin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/singles')
    df = sin_df.iloc[:,:441].to_numpy() #np.zeros()
    labels = sin_df['spectral_type']

    temp = []
    for i in range(len(labels)):
        temp.append([labels[i]])
    labels = np.array(temp)

    shuffler = np.arange(np.shape(df)[0])
    np.random.shuffle(shuffler)
    shuffled_images = df[shuffler], labels[shuffler]
    
    return shuffled_images


