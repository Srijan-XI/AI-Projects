#movie_recommender/src/main.py
from preprocess import load_and_preprocess
from recommender import get_recommendations

if __name__ == "__main__":
    # Load data & preprocess
    df, tfidf_matrix = load_and_preprocess("data/movies.csv")

    # User input
    movie_name = input("Enter a movie title: ")

    # Get recommendations
    recommendations = get_recommendations(movie_name, df, tfidf_matrix)

    print("\nTop 5 Recommendations:")
    for i, rec in enumerate(recommendations, start=1):
        print(f"{i}. {rec}")