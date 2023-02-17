
def clean_title(df):
    df = df.drop(df[df['title'].str.len() < 3].index)
    return df