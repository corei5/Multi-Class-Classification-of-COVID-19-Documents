import pandas as pd

def remove_nan(df):

    df = df[pd.notnull(df['text'])]
    df = df[pd.notnull(df['authors'])]
    df = df[pd.notnull(df['title'])]
    df = df[pd.notnull(df['publish_time'])]
    df = df[pd.notnull(df['journal'])]
    df = df[pd.notnull(df['tag_target_class'])]


    return df
