

import warnings
warnings.filterwarnings('ignore')

from dataloader import load_data,explore_data,create_user_item_matrix
from model import SVDModel
from recommender import MovieRecommender
from evaluation import evaluate_model

def main():

    print("\n" + "="*70)
    print("MOVIE RECOMMENDATION SYSTEM USING SVD")
    print("="*70)

    # --- 1. Load and Prepare Data ---
    ratings, movies = load_data()
    explore_data(ratings)
    user_item_matrix_filled, user_item_matrix = create_user_item_matrix(ratings)

    # --- 2. Train SVD Model ---
    svd_model = SVDModel()
    svd_model.fit(user_item_matrix_filled)


    predictions = svd_model.predict()


    recommender = MovieRecommender(
        predictions_matrix=predictions,
        user_item_matrix=user_item_matrix,
        movies_df=movies,
        vt_matrix=svd_model.Vt # Pass Vt for item similarity
    )


    recommender.recommend_movies(user_id=2, num_movies=5)
    recommender.recommend_movies(user_id=5, num_movies=5)


    recommender.find_similar_movies(movie_id=3, num_similar=5)
    recommender.find_similar_movies(movie_id=1, num_similar=5)


    rmse, mae = evaluate_model(predictions, user_item_matrix)


    print("\n" + "="*70)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nKey Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()