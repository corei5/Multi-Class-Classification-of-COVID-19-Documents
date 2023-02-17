import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Reduction of BOW and TF-IDF by parameter min_df
MIN_DF = 32

def author_names_matrix(df):

    cvec_authors = CountVectorizer(analyzer = "word", tokenizer = None, ngram_range=(1,50), min_df=MIN_DF, lowercase = True,strip_accents = "ascii", binary= True,stop_words='english')
    matrix_authors = cvec_authors.fit_transform(df['authors'])
    tokens = cvec_authors.get_feature_names()
    matrix_authors_pd=pd.DataFrame(data=matrix_authors.toarray(), index=df.index,columns=tokens)

    vect_authors_tfidf = TfidfVectorizer( tokenizer = None, ngram_range=(1,50), min_df=MIN_DF, lowercase = True,strip_accents = "ascii",stop_words='english')
    matrix_authors_tfidf= vect_authors_tfidf.fit_transform(df['authors'])
    tokens_authors_tfidf = vect_authors_tfidf.get_feature_names()
    matrix_authors_tfidf_pd=pd.DataFrame(data=matrix_authors_tfidf.toarray(), index=df.index,columns=tokens_authors_tfidf)

    return matrix_authors_pd, matrix_authors_tfidf_pd
