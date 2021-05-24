
import seaborn as sns
import os
from pydataset import data
from scipy import stats
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from acquire import get_titanic_data, get_iris_data

# ************************************ IRIS DATA ***********************************

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





def iris_split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.species)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.species)
    return train, validate, test



# ************************************ TITANIC DATA***********************************

def clean_data(df, dummies):
    '''
    This function will drop any duplicate observations, 
    drop ['deck', 'embark_town', 'class'], fill missing embarked with 'Southampton'
    and create dummy vars from sex and embarked. 
    '''

    #clean data
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embark_town', 'class'])
    df['embarked'] = df.embarked.fillna(value='S')
    #create a dummy df
    dummy_df = pd.get_dummies(df[dummies], drop_first=[True, True])
    ## Concatenate the dummy_df dataframe above with the original df
    df = pd.concat([df, dummy_df], axis=1)
    # drop the columns that we already use to create dummy_df
    df = df.drop(columns= dummies)
    
    return df



def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on survived.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.survived)
    return train, validate, test

def impute_mode(train, validate, test, column , method):
    '''
    impute a choosen strategy (method) for age column into observations with missing values.
    column:  is the column to impute or fill the missing values in and 
    method:  is the type of strategy(media, media, most_frequent)
    '''
    imputer = SimpleImputer(strategy= method, missing_values= np.nan)
    train[[column]] = imputer.fit_transform(train[[column]])
    validate[[column]] = imputer.transform(validate[[column]])
    test[[column]] = imputer.transform(test[[column]])
    return train, validate, test


def prep_titanic_data(df, column, method ,dummies):
    '''
    takes in a dataframe of the titanic dataset that was  acquired before and returns a cleaned dataframe
    arguments:
    - df: a pandas DataFrame with the expected feature names and columns
    - column : the name of the column to fill or impute the missing values in
    - method: type of strategy (median, mean, most_frequent) for SimpleImputer
    - dummies: list of 2 columns to create a dummy variable 
    return: 
    train, validate, test (three dataframes with the cleaning operations performed on them)
    Example :
    train, validate, test = prepare.prep_titanic_data(df, column = 'age', method = 'median', dummies = ['embarked', 'sex'])
    '''
    #clean data
    df = clean_data(df, dummies)
    
    # split data into train, validate, test dfs
    train, validate, test = split_data(df)

    # impute the chosen strategy (median)  for  the selected column (age) into null values in age column
    train, validate, test = impute_mode(train, validate, test, column , method)
   
    return train, validate, test