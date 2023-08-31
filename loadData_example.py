#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sem Hoogteijling

"""

import pandas as pd


def loadSpectralFeatures():
    
    path_to_data_folder = '/path/to/data/folder/' #specify path to data folder
    
    Xy_test = pd.read_csv(path_to_data_folder+'Xy_test.csv')
    Xy_train = pd.read_csv(path_to_data_folder+'Xy_train.csv')
    
    y_train = Xy_train['label']
    X_train = Xy_train.drop(columns = 'label')
    y_test = Xy_test['label']
    X_test = Xy_test.drop(columns = 'label')
    
    return X_train, y_train, X_test, y_test
