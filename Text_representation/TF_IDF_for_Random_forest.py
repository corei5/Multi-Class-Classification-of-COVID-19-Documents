import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_DF = 32

#MIN_DF = 5

def tf_idf_for_random_forest(df, col_name):
    vect_tfidf = TfidfVectorizer(analyzer = "word",
                           tokenizer = None,
                           ngram_range=(1,3),
                           lowercase = True,
                           strip_accents = "ascii",
                           binary= True,
                           stop_words='english',
                           min_df=MIN_DF)

    matrix_tfidf= vect_tfidf.fit_transform(df[col_name])
    tokens_tfidf = vect_tfidf.get_feature_names()
    matrix_tfidf_pd = pd.DataFrame(data=matrix_tfidf.toarray(), index=df.index,columns=tokens_tfidf)
    return matrix_tfidf_pd, matrix_tfidf, vect_tfidf

