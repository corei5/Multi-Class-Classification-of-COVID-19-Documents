import sys
import numpy as np
sys.setrecursionlimit(100000)

def age_of_publications(df):
    #df['year'] = np.nan
    #df[['year']] = df['published_year'].str.split('-', 1, expand=True)[0]
    #df = df[df['year'].notna()]
    #df = df[df['year'].apply(lambda x: str(x).isdigit())]
    #df['year'] = df['year'].apply(lambda x: int(x))
    df['age'] = 2021 - df['published_year']
    return df