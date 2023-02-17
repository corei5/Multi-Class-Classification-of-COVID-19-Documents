import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

MIN_DF = 32

def bow_matrix_for_random_forest(df, col_name):
    cvec = CountVectorizer(analyzer = "word",
                           tokenizer = None,
                           ngram_range=(1,3),
                           lowercase = True,
                           strip_accents = "ascii",
                           binary= True,
                           stop_words='english',
                           min_df=MIN_DF)

    matrix_bow = cvec.fit_transform(df[col_name])
    tokens = cvec.get_feature_names()
    matrix_bow_pd = pd.DataFrame(data=matrix_bow.toarray(), index=df.index,columns=tokens)
    return matrix_bow_pd, matrix_bow, cvec