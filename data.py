import numpy as np 
import pandas as pd

def get_data():
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



