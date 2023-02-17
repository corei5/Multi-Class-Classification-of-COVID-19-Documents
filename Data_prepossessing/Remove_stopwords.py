import nltk.corpus
# nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def remove_stopwords(df):
    df['text'] =  df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df['title'] = df['title'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop_words)]))
    return df

