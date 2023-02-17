
def bibliometric_features(df):
    bibliometric_features = df.reset_index()[[  "journal", "author_count", "license", "age"]]  # ,,"pubmed_id", "source_x",, "tag_disease_corona"
    #bibliometric_features['pubmed_id'] = bibliometric_features['pubmed_id'].apply(lambda x: str(x).split('/')[len(str(x).split('/')) - 1])
    
    #bibliometric_features['pubmed_id']=bibliometric_features['pubmed_id'].astype(int)
    
    #bibliometric_features['pubmed_id'] = bibliometric_features['pubmed_id'].astype(str)
    
    #bibliometric_features=bibliometric_features.merge(impacts_ais_jcats_df2,on='pubmed_id', how='left').dropna().set_index("doi")
    
    #del bibliometric_features['pubmed_id']
    
    return bibliometric_features