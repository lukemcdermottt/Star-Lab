import numpy as np 
import pandas as pd

def get_data(path):
    #Load Dataframes
    bin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/Star-Lab/spectral_templates_data_version_june20.h5', key = '/binaries')
    sin_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/Star-Lab/spectral_templates_data_version_june20.h5', key = '/singles')
    #wav_df = pd.read_hdf('/Users/lukemcdermott/Desktop/Physics/spectral_templates_data_version_june20.h5', key = '/wavegrid')

    #Prep data for NN with labels: bin ex = [flux][0], sin ex = [flux][1]
    bin_data = np.array([bin_df.iloc[:,:441].to_numpy(), np.zeros(bin_df.shape[0])])
    sin_data = np.array([sin_df.iloc[:,:441].to_numpy(), np.ones(sin_df.shape[0])])

    data = bin_data
    for _ in range (90):
        data = np.concatenate((data, sin_data), axis = 0)

    #data = [inputs, targets]
    #inputs = [[flux arr], [flux arr], ...,[flux arr]]
    #targets = [1, 0, ..., 1]

    #shuffle data
    shuffler = np.random.permutation(len(data[0]))
    shuffled_data = np.array([ data[0][shuffler], data[1][shuffler] ])
    
    return shuffled_data




