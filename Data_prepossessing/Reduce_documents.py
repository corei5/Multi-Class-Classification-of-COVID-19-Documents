
def reduce_documents(bibliometric_features,df):
    df = df.reset_index()[["doi"]].set_index("doi")
    new_ids = bibliometric_features.reset_index()[["doi"]].set_index("doi")
    df = new_ids.merge(df, how='inner', left_index=True, right_index=True)
    return new_ids, df