from env import username, host, password
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


################################## Acquire CC csv Function ############################


def wrangle_cc():
    '''
    This function reads in card_transdata data from Kaggle, writes data to
    a csv file if a local file does not exist, and returns a df
    '''
    filename = 'card_transdata.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        
    return df

################################## Overview Function ############################


def overview(df):
    '''
    This function prints shape of DataFrame, .info, and .describe
    '''
    print('--- Shape: {}'.format(df.shape))
    print('____________________________________________________')
    print('--- Info')
    df.info()
    print('____________________________________________________')
    print('--- Column Descriptions')
    print()
    print(df.describe())
    
    
    ################################## Outlier Function ############################
    
def prep_cc(df):
    for cols in df:

        q1, q3 = df[cols].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        k=1.5
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[cols] > lower_bound) & (df[cols] < upper_bound)]
        
        return df
    
    
    ################################## Split Function ############################


def split_data(df, target):
    '''
    This function take in a dataframe performs a train, validate, test split
    Returns train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test
    and prints out the shape of train, validate, test
    '''
    
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)

    #Split into X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    # Have function print datasets shape
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
   
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


################################## Scale Function ############################


def scale_data(train, 
               validate, 
               test, 
               columns_to_scale, return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns them scaled.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of the original data so no leakage
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # variable 
    mm_scaler = MinMaxScaler()
    # fit it to scaler
    mm_scaler.fit(train[columns_to_scale])
    # scaling train, validate, test, and columns
    train_scaled[columns_to_scale] = pd.DataFrame(mm_scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(mm_scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(mm_scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return mm_scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
