
def author_count(df):
    df['author_count'] = df['authors'].apply(lambda x: len(x.split(",")))
    return df