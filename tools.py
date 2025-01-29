import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def filter_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    stop = set(stopwords.words('english'))
    text = ' '.join([palavra for palavra in text.split() if palavra not in stop])
    return text

def hot_encoding(columns, df):
    return pd.get_dummies(df, columns = columns, dtype = int)


def target_encoding(source_column, target_column, df):
    avg = df.groupby(source_column)[target_column].mean()
    n_colum = source_column + '_encoded'
    df[n_colum] = df[source_column].map(avg)
    df.drop(columns = source_column, inplace = True)
    return df

def tf_idf(df, column):
    tfidf = TfidfVectorizer(max_features=76)
    X_tfidf = tfidf.fit_transform(df[column])

    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out(), index=df.index)

    return tfidf_df