import math
import numpy as np
import pandas as pd
import os
from mxnet import nd, npx
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.dataset import ArrayDataset


def data_cleaning(data):
  
  infectious_data_path = os.path.join(os.getcwd(), data["infectious_data_path"])
  deceased_data_path = os.path.join(os.getcwd(), data["deceased_data_path"])
  recovered_data_path = os.path.join(os.getcwd(), data["recovered_data_path"])

  infectious_data = pd.read_csv(infectious_data_path)
  deceased_data = pd.read_csv(deceased_data_path)
  recovered_data = pd.read_csv(recovered_data_path)

  infectious_data = infectious_data.groupby('Country/Region').sum()
  deceased_data = deceased_data.groupby('Country/Region').sum()
  recovered_data = recovered_data.groupby('Country/Region').sum()
  
  infectious_data = infectious_data[infectious_data.index == data["select_country"]].drop(columns = ['Lat', 'Long', 'Province/State'])
  deceased_data = deceased_data[deceased_data.index == data["select_country"]].drop(columns = ['Lat', 'Long', 'Province/State'])
  recovered_data = recovered_data[recovered_data.index == data["select_country"]].drop(columns = ['Lat', 'Long', 'Province/State'])

  infectious_data = infectious_data.T
  infectious_data.columns = ['infectious']
  infectious_data.index.rename('date', inplace=True)
  infectious_data.index = pd.to_datetime(infectious_data.index, utc=True, format = '%m/%d/%Y')

  deceased_data = deceased_data.T
  deceased_data.columns = ['deceased']
  deceased_data.index.rename('date', inplace=True)
  deceased_data.index = pd.to_datetime(deceased_data.index, utc=True, format ='%m/%d/%y')

  recovered_data = recovered_data.T
  recovered_data.columns = ['recovered']
  recovered_data.index.rename('date', inplace=True)
  recovered_data.index = pd.to_datetime(recovered_data.index, utc=True, format ='%m/%d/%y')
  
  infectious_array = infectious_data['infectious'].to_numpy()
  recovered_array = recovered_data['recovered'].to_numpy()
  deceased_array = deceased_data['deceased'].to_numpy()
  
  susceptible_array = np.subtract(np.subtract(np.subtract(data["select_country_population"] 
                                                          * np.ones(shape = infectious_array.shape), 
                                                          infectious_array), 
                                                          deceased_array),
                                                          recovered_array).astype('int')
  susceptible_data = pd.DataFrame(data = susceptible_array, index = infectious_data.index, columns=['susceptible'])
  
  clean_data = susceptible_data.join([infectious_data, recovered_data, deceased_data], how = 'left')
  clean_data = clean_data[clean_data.infectious >= 1]
  dif_clean_data = clean_data.diff(periods = 1, axis = 0).dropna(inplace = False)
  clean_data = clean_data.join(dif_clean_data, how = 'left', lsuffix='_cummulative', rsuffix='_difference')
  clean_data.fillna(value = 0, inplace=True)
  clean_data = clean_data / 1E6

  return clean_data


def data_preprocessing(x, y, window_size):

  num_to_unpack = math.floor(x.shape[0] / window_size)
  start_idx = x.shape[0] - num_to_unpack * window_size
  x = x[start_idx:]
  y = y[start_idx:]

  x = np.expand_dims(x, axis = 1)
  x = np.split(x, x.shape[0]/window_size, axis = 0)
  x = np.concatenate(x, axis = 1)
  x = np.transpose(x, axes = (1, 0, 2))
  y = y[::window_size]

  x = nd.array(x)
  y = nd.array(y)

  dataset = ArrayDataset(x, y)

  return dataset, start_idx


def data_preparation(configs):

    clean_data = data_cleaning(configs["data"])

    train_test_split = configs["train"]["train_test_split"]
    train_split = int(math.floor(clean_data.shape[0] * train_test_split[0]))
    val_split = int(math.floor(clean_data.shape[0] * (train_test_split[0] + train_test_split[1])))
    prepared_train_data = clean_data[:train_split]
    prepared_val_data = clean_data[train_split:val_split]
    prepared_test_data = clean_data[val_split:]
    time_stamps = clean_data.index
    time_stamps_train = time_stamps[:train_split]
    time_stamps_val = time_stamps[train_split:val_split]
    time_stamps_test = time_stamps[val_split:]

    X_train = prepared_train_data[:(prepared_train_data.shape[0] - configs["model"]["window_size"])].values
    Y_train = prepared_train_data[configs["model"]["window_size"]:][['susceptible_cummulative', 'infectious_cummulative', 'recovered_cummulative', 'deceased_cummulative']].values
    time_stamps_train = time_stamps_train[:(time_stamps_train.shape[0] - configs["model"]["window_size"])]

    data_train, start_idx_train = data_preprocessing(X_train, Y_train, configs["model"]["window_size"])

    X_val = prepared_val_data[:(prepared_val_data.shape[0] - configs["model"]["window_size"])].values
    Y_val = prepared_val_data[configs["model"]["window_size"]:][['susceptible_cummulative', 'infectious_cummulative', 'recovered_cummulative', 'deceased_cummulative']].values
    time_stamps_val = time_stamps_val[:(time_stamps_val.shape[0] - configs["model"]["window_size"])]
    data_val, start_idx_val = data_preprocessing(X_val, Y_val, configs["model"]["window_size"])

    X_test = prepared_test_data[:(prepared_test_data.shape[0]- configs["model"]["window_size"])].values
    Y_test = prepared_test_data[configs["model"]["window_size"]:][['susceptible_cummulative', 'infectious_cummulative', 'recovered_cummulative', 'deceased_cummulative']].values
    time_stamps_test = time_stamps_test[:(time_stamps_test.shape[0] - configs["model"]["window_size"])]
    data_test, start_idx_test = data_preprocessing(X_test, Y_test, configs["model"]["window_size"])

    bool_device = True if npx.num_gpus() > 0 else False

    data_train_ld = DataLoader(dataset = data_train, batch_size=configs["train"]["batch_size"], pin_memory=bool_device)
    data_val_ld = DataLoader(dataset = data_val, batch_size=configs["train"]["batch_size"], pin_memory=bool_device)
    data_test_ld = DataLoader(dataset = data_test, batch_size=1, pin_memory=bool_device)

    return ((data_train_ld, data_val_ld, data_test_ld), (start_idx_train, start_idx_val, start_idx_test), (time_stamps_train, time_stamps_val, time_stamps_test))