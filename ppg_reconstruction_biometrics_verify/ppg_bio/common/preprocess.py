from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
from numpy import array, split

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
from numpy import array, split
from sklearn.utils import shuffle
#from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
import heartpy as hp
import pickle
import pandas as pd
import os
import json

from pathlib import Path

def SamplePerClass_Index(dataset_array, num_samples):
    ''' Takes the data and number of samples 
    to return the index of equall number of samples per each value in the data'''
    
    df = pd.DataFrame(dataset_array)
    np.random.seed(42)
    df = df.groupby(df[0]).apply(lambda s: s.sample(num_samples)).droplevel(level=0)
    np.random.seed(42)
    indexes = np.array((df.index))
    return indexes



# split the signal
def split_dataset(data, n_input):
    ''' Takes the signal data with the number of data point for windows 
    and returns the split non-overlapped windows '''

    remainder = len(data) % n_input
    if remainder != 0:
        data = data[:-remainder]
    data = array(split(data, len(data)/n_input))
    return data


#Creating the inputs based on the overlapping / This finction givs us two same windows as x and y
def to_supervised(data, n_input, shift ):
    ''' Takes the split non-overlapped windows with the number of data point
    and number of data point for overlapping which returns the overlapping windows'''

    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))  #flatten the split non-overlapped data
    X = list()
    in_start = 0
    #step over the entire history one time step at a time
    for i in range(len(data)):
        #define the end of the input sequence
        in_end = in_start + n_input
        # ensure we have enough data for this instance
        if in_end <= len(data):
            x_input = data[in_start:in_end]
            X.append(x_input)
        # move along one time step
        in_start += shift
    X = array(X)
    return X


def all_equal(data):
    ''' takes the windows and check if all datapoint are equal to return True '''
    
    iterator = iter(data)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)



def label_maker_Dalia(data_ppg, data_activity):
    labels_activity = []
    Not_Equal_id = []
    for i in range(data_activity.shape[0]):
        Check = all_equal(data_activity[i,:])
        if Check == True :
            labels_activity.append(data_activity[i,:][0])
        else:
            Not_Equal_id.append(i)
    
    Final_data_ppg = np.delete(data_ppg, Not_Equal_id, axis=0)
    labels_activity = (np.array(labels_activity)).reshape(-1,1)
    return Final_data_ppg, labels_activity



def load_and_preprocess_data(config, id_data, scaler=None, fit_scaler=True, removing=True, is_test = True):

    """
    Load and preprocess signal data from a given file path.

    Parameters:
    - config: config file including all parameters.
    - id_data: test person id. 
    - scaler: Instance of a scaler (e.g., StandardScaler). If None, a new scaler will be created.
    - fit_scaler: Boolean flag indicating whether to fit the scaler to the data or use it for transformation only.

    Returns:
    - Preprocessed signal data.
    - Scaler used for the data normalization.
    """
    params = config["prepare_datasets_params"]
    n_input = params["n_input"]
    shift = params["shift"]
    signal_freq = params["signal_freq"]
    low_freq = params["low_freq"]
    high_freq = params["high_freq"]
    filter_order = params["filter_order"]
    
    data_directory = config.get("data_directory", "")
    data_path = os.path.join(data_directory, f'S{id_data}.pkl')

    
    try:
        with open(data_path, 'rb') as handle:
            data_dic = pickle.load(handle, encoding='latin1')
        
        print(data_path)
        dataset = data_dic['signal']['wrist']['BVP'].astype('float32') 
         
        filtered_data = hp.filter_signal(dataset[:,0], cutoff=[low_freq, high_freq], sample_rate=signal_freq, order=filter_order, filtertype='bandpass') 
        filtered_data_train, filtered_data_test = train_test_split(filtered_data, test_size=0.2, shuffle=False)  

        if scaler is None:
            scaler = StandardScaler()
        
        if fit_scaler:
            normalized_data_train = scaler.fit_transform(filtered_data_train.reshape(-1, 1))
        else:
            normalized_data_train = scaler.transform(filtered_data_train.reshape(-1, 1))

        split_data_ppg_train = split_dataset(normalized_data_train, n_input) 
        x_train = to_supervised(split_data_ppg_train, n_input , shift) 

        normalized_data_test = scaler.transform(filtered_data_test.reshape(-1, 1))
        split_data_ppg_test= split_dataset(normalized_data_test, n_input) 
        x_test = to_supervised(split_data_ppg_test, n_input , shift) 

        
        return x_train, x_test
    
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None, None, None




def prepare_datasets(config, persons):
    """
    Prepare training and testing datasets.
    """

    save_dir = Path(config["dataset"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for id_data in persons:
        train_x, test_x = load_and_preprocess_data(
            config, id_data, scaler=None, fit_scaler=True, is_test=False
        )

        np.savez(
            save_dir / f"{id_data}.npz",
            train_data=train_x,
            label_data=test_x
        )


# Load configuration
with open('ppg-dalia_config.json') as config_file:
    config = json.load(config_file)

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#########################

# Load data using parameters from config
list_persons = config["list_persons"]

prepare_datasets(config, list_persons)

import sys
sys.path.append('./') 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import mne
import numpy as np
from numpy import array, split
from sklearn.utils import shuffle
#from tensorflow.keras.utils import to_categorical
#import heartpy as hp
import pandas as pd
import os
import json
from ltbio.biosignals.modalities import PPG
from ltbio.processing.filters import *



def SamplePerClass_Index(dataset_array, num_samples):
    ''' Takes the data and number of samples 
    to return the index of equall number of samples per each value in the data'''
    
    df = pd.DataFrame(dataset_array)
    np.random.seed(42)
    df = df.groupby(df[0]).apply(lambda s: s.sample(num_samples)).droplevel(level=0)
    np.random.seed(42)
    indexes = np.array((df.index))
    return indexes



# split the signal
def split_dataset(data, n_input):
    ''' Takes the signal data with the number of data point for windows 
    and returns the split non-overlapped windows '''

    remainder = len(data) % n_input
    if remainder != 0:
        data = data[:-remainder]
    data = array(split(data, len(data)/n_input))
    return data


#Creating the inputs based on the overlapping / This finction givs us two same windows as x and y
def to_supervised(data, n_input, shift ):
    ''' Takes the split non-overlapped windows with the number of data point
    and number of data point for overlapping which returns the overlapping windows'''

    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))  #flatten the split non-overlapped data
    X = list()
    in_start = 0
    #step over the entire history one time step at a time
    for i in range(len(data)):
        #define the end of the input sequence
        in_end = in_start + n_input
        # ensure we have enough data for this instance
        if in_end <= len(data):
            x_input = data[in_start:in_end]
            X.append(x_input)
        # move along one time step
        in_start += shift
    X = array(X)
    return X


def all_equal(data):
    ''' takes the windows and check if all datapoint are equal to return True '''
    
    iterator = iter(data)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)



def label_maker_Dalia(data_ppg, data_activity):
    ''' Takes both ppg and activity window data to label the activity signal / 
    This function removes the activity window with not equal data point in the window
    besides the corosponding ppg window data and label the ppg window based on the activity window '''

    labels_activity = []
    Not_Equal_id = []
    for i in range(data_activity.shape[0]):
        Check = all_equal(data_activity[i,:])
        if Check == True :
            labels_activity.append(data_activity[i,:][0])
        else:
            Not_Equal_id.append(i)
    
    Final_data_ppg = np.delete(data_ppg, Not_Equal_id, axis=0)
    labels_activity = (np.array(labels_activity)).reshape(-1,1)
    return Final_data_ppg, labels_activity

def extract_clean_segments(data, nan_start, nan_end, n_input, shift):
    parts = []
    for part in [data[:nan_start], data[nan_end:]]:
        if len(part) >= n_input:
            normalized = part.reshape(-1, 1)
            split = split_dataset(normalized, n_input)
            windows = to_supervised(split, n_input, shift)
            parts.append(windows)
    if parts:
        return np.concatenate(parts, axis=0)
    else:
        return np.empty((0, n_input, 1))


def load_and_preprocess_data(config, id_data, scaler=None, fit_scaler=True, removing=True, is_test = True):

    """
    Load and preprocess signal data from a given file path.

    Parameters:
    - config: config file including all parameters.
    - id_data: test person id. 
    - scaler: Instance of a scaler (e.g., StandardScaler). If None, a new scaler will be created.
    - fit_scaler: Boolean flag indicating whether to fit the scaler to the data or use it for transformation only.

    Returns:
    - Preprocessed signal data.
    - Scaler used for the data normalization.
    """
    params = config["prepare_datasets_params"]
    n_input = params["n_input"]
    shift = params["shift"]
    signal_freq = params["signal_freq"]
    low_freq = params["low_freq"]
    high_freq = params["high_freq"]
    filter_order = params["filter_order"]
    
    data_directory = config.get("data_directory", "")
    data_path = os.path.join( f'{data_directory}/{id_data}', f'ppg.biosignal')
    try:
        print(data_path)
        ppg = PPG.load(data_path)
        #print(ppg)
        if(id_data == "ME93"):
            left_wrist_ppg = ppg
        else:
            left_wrist_ppg = ppg['Left Wrist']
       
        passband_filter = FrequencyDomainFilter(FrequencyResponse.FIR, BandType.BANDPASS, (low_freq, high_freq), order=filter_order)  # design the filter
        filtered_data = passband_filter(left_wrist_ppg).to_array().squeeze()

        if np.isnan(filtered_data).any():
            nan_mask = np.isnan(filtered_data)
            splits = np.diff(np.where(np.concatenate(([nan_mask[0]], nan_mask[:-1] != nan_mask[1:], [True])))[0])
            starts = np.where(np.concatenate(([nan_mask[0]], nan_mask[:-1] != nan_mask[1:])))[0][::2]

            nan_lengths = splits[::2]
            max_nan_idx = np.argmax(nan_lengths)
            nan_start_pos = starts[max_nan_idx]
            nan_end_pos = nan_start_pos + nan_lengths[max_nan_idx]

            x_train = extract_clean_segments(filtered_data, nan_start_pos, nan_end_pos, n_input, shift)
            if len(x_train) == 0:
                print(f"⚠️ Missing data {id_data}")
                return None, None

            # ✅ Fit scaler
            if scaler is None:
                scaler = StandardScaler()
            x_train_flat = x_train.reshape(-1, 1)
            x_train_flat = scaler.fit_transform(x_train_flat)
            x_train = x_train_flat.reshape(x_train.shape)

           
            filtered_data_test = filtered_data[nan_end_pos:]
            if len(filtered_data_test) < n_input:
                print(f"⚠️ Insufficient data {id_data}")
                return None, None

            normalized_data_test = scaler.transform(filtered_data_test.reshape(-1, 1))
            split_data_ppg_test = split_dataset(normalized_data_test, n_input)
            x_test = to_supervised(split_data_ppg_test, n_input, shift)

        else:
            filtered_data_train, filtered_data_test = train_test_split(filtered_data, test_size=0.2, shuffle=False)

            if scaler is None:
                scaler = StandardScaler()
            normalized_data_train = scaler.fit_transform(filtered_data_train.reshape(-1, 1))
            split_data_ppg_train = split_dataset(normalized_data_train, n_input)
            x_train = to_supervised(split_data_ppg_train, n_input , shift)

            normalized_data_test = scaler.transform(filtered_data_test.reshape(-1, 1))
            split_data_ppg_test = split_dataset(normalized_data_test, n_input)
            x_test = to_supervised(split_data_ppg_test, n_input , shift)

        filtered_data_train, filtered_data_test = train_test_split(filtered_data, test_size=0.2, shuffle=False)
        normalized_data_test = scaler.transform(filtered_data_test.reshape(-1, 1))
        split_data_ppg_test= split_dataset(normalized_data_test, n_input)
        x_test = to_supervised(split_data_ppg_test, n_input , shift)

        
        return x_train, x_test
    
    except Exception as e:
        print(f"Error loading and preprocessing data: {e}")
        return None, None




def prepare_datasets(config, persons):
    """
    Prepare training and testing datasets.
    """

    save_dir = Path(config["dataset"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for id_data in persons:
        train_x, test_x = load_and_preprocess_data(
            config, id_data, scaler=None, fit_scaler=True, is_test=False
        )

        np.savez(
            save_dir / f"{id_data}.npz",
            train_data=train_x,
            label_data=test_x
        )

# Load configuration
with open('scientisst_config.json') as config_file:
    config = json.load(config_file)

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load data using parameters from config
list_persons = config["list_persons"]

prepare_datasets(config, list_persons)
