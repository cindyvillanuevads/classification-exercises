import numpy as np
import pandas as pd
import seaborn as sns
import os
from pydataset import data
from scipy import stats


def prep_iris(df):
    '''
    takes in a iris df that was  acquired before  and return a cleaned
    dataframe: dropeed species_id and measurement_id columns, renamed species_name
    and created dummy variables of the species name
    '''
    #drop columns
    df = df.drop(columns= ['species_id','measurement_id'])
    #rename columns
    df.rename(columns={"species_name": "species"}, inplace =True)
    #create dummy variables of the species name
    dummy_df = pd.get_dummies(df[['species']], dummy_na=False, drop_first=[True])
    #  concat dummy_df with my df
    df = pd.concat([df, dummy_df], axis =1)
    return df
    
    