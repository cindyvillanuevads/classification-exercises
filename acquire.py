
import pandas as pd
import numpy as np
import os

# visualize
import seaborn as sns


# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

# acquire
from pydataset import data

# *************************************  connection url **********************************************

# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    from env import host, username, password
    return f'mysql+pymysql://{username}:{password}@{host}/{db_name}'

# *************************************  TITANIC DB **********************************************


# Use the above helper function and a sql query in a single function.
def get_new_titanic_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = 'SELECT * FROM passengers'
    return pd.read_sql(sql_query, get_connection('titanic_db'))



def get_titanic_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('titanic_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_new_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
    return df

# *************************************  IRIS DB **********************************************

# Use helper function and sql query 

def get_new_iris_data():
    '''
    This function reads in the iris data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
    SELECT *
    FROM measurements
    JOIN species USING (species_id)
    '''
    return pd.read_sql(sql_query, get_connection('iris_db'))





def get_iris_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('iris_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('iris_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_new_iris_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('iris_df.csv')
        
    return df