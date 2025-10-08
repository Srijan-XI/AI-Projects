#movie_recommender/src/preprocess.py
import pandas as pd
import os

def load_and_preprocess(file_path):
    # Get absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, file_path)

    df = pd.read_csv(full_path)
    df['overview'] = df['overview'].fillna('')

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    return df, tfidf_matrix
