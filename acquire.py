from env import username, host, password
import os
import pandas as pd
import numpy as np



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

