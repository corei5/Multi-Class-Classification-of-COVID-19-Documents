
def clean_abstract(df):
    df = df.drop(df[df['text'].str.len() < 3].index)
    return df