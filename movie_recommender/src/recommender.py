#movie_recommender/src/preprocess.py
import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(title, df, tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Map indices to movie titles
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    if title not in indices:
        return [f"Movie '{title}' not found in dataset."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 excluding itself
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices].tolist()