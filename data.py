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
    print(type(data))
    labels = np.zeros(bin_data.size)
    for _ in range (90):
        data = np.concatenate((data, sin_data), axis = 0)
        labels = np.concatenate((labels, np.ones(sin_data.size)), axis = 0)
    

    #shuffle data
    arr = np.arange(10)
    np.random.shuffle(arr)

    print(np.shape(data))
    #use length of data -> i think data.size might be wrong above
    shuffler = np.arange(20)
    np.random.shuffle(shuffler)
    print(shuffler)
    shuffled_data = data[shuffler] #, labels[shuffler] ])
    
    print('data done')
    return shuffled_data




