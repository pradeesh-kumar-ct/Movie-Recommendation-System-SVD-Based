

import pandas as pd
import numpy as np


class MovieRecommender:


    def __init__(self, predictions_matrix, user_item_matrix, movies_df, vt_matrix):
        self.predictions = predictions_matrix
        self.user_item_matrix = user_item_matrix
        self.movies = movies_df
        self.Vt = vt_matrix  #

    def recommend_movies(self, user_id, num_movies=5):


        user_idx = user_id - 1
        if not (0 <= user_idx < self.predictions.shape[0]):
            print(f"Error: User ID {user_id} is out of bounds.")
            return pd.DataFrame()

        user_predictions = self.predictions[user_idx, :]
        user_ratings = self.user_item_matrix.loc[user_id]

        # Get IDs of movies the user has already rated
        already_rated_indices = user_ratings[user_ratings > 0].index.tolist()

        # Create a DataFrame of all movies and their predicted ratings
        recommendations = pd.DataFrame({
            "movieId": self.user_item_matrix.columns,
            "predicted_rating": user_predictions,
        })

        # Filter OUT movies the user has already rated
        recommendations = recommendations[~recommendations['movieId'].isin(already_rated_indices)]

        # Sort by predicted rating and get the top N
        top_recommendations = recommendations.sort_values(
            by='predicted_rating', ascending=False
        ).head(num_movies)

        # Merge with movie titles for display
        top_recommendations = top_recommendations.merge(self.movies, on='movieId')

        print(f"\n--- Top {num_movies} Recommendations for User {user_id} ---")
        for idx, row in top_recommendations.iterrows():
            print(f"Title: {row['title']:<50} | Predicted Rating: {row['predicted_rating']:.2f}")

        return top_recommendations

    def find_similar_movies(self, movie_id, num_similar=5):

        if movie_id not in self.user_item_matrix.columns:
            print(f"Error: Movie ID {movie_id} not found.")
            return


        movie_idx = self.user_item_matrix.columns.get_loc(movie_id)


        movie_feature_vector = self.Vt[:, movie_idx]


        similarities = np.dot(self.Vt.T, movie_feature_vector) / (
                np.linalg.norm(self.Vt.T, axis=1) * np.linalg.norm(movie_feature_vector) + 1e-9
        )


        similar_indices = np.argsort(similarities)[::-1][1:num_similar + 1]

        original_title = self.movies[self.movies['movieId'] == movie_id]['title'].values[0]
        print(f"\n--- Movies Similar to '{original_title}' ---")

        similar_movies = []
        for idx in similar_indices:
            similar_movie_id = self.user_item_matrix.columns[idx]
            title = self.movies[self.movies['movieId'] == similar_movie_id]['title'].values[0]
            similarity_score = similarities[idx]
            print(f"Title: {title:<50} | Similarity: {similarity_score:.2f}")
            similar_movies.append((similar_movie_id, title, similarity_score))

        return similar_movies