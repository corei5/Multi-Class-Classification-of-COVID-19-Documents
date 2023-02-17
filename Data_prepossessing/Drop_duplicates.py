

def drop_duplicates(df):
    df = df.drop_duplicates(subset=['pubmed_id'], keep=False)
    df = df.drop_duplicates(subset=['doi', 'journal'], keep=False)
    return df