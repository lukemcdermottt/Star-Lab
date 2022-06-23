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
    labels = np.empty(0)
    for i in bin_data:
        labels = np.concatenate((labels, np.array([1,0])), axis = 0)
    for i in sin_data:
        sin_labels = np.concatenate((labels, np.array([0,1])), axis = 0)
    for _ in range (90):
        data = np.concatenate((data, sin_data), axis = 0)
        labels = np.concatenate((labels, sin_labels), axis = 0)
    
    #shuffle data
    shuffler = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffler)
    shuffled_data = data[shuffler], labels[shuffler]

    
    return shuffled_data




