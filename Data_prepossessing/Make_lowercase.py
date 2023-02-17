
def make_lowercase(df):
    df["text"] = df["text"].str.lower().replace('text', '')
    df["title"] = df["title"].str.lower().replace('title', '')
    return df