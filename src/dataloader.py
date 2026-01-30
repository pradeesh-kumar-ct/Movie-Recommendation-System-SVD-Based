import pandas as pd
import numpy as np
from config import RATINGS_FILE, MOVIES_FILE


def load_data():

    try:
        ratings = pd.read_csv(RATINGS_FILE)
        movies = pd.read_csv(MOVIES_FILE)
        print("\n" + "=" * 70)
        print("Data Loaded Successfully")
        print("=" * 70)
        print(f"Loaded {len(ratings):,} ratings from {ratings['userId'].nunique()} users.")
        print(f"Loaded {len(movies)} movies.")
        return ratings, movies
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please ensure 'u.csv' and 'mm.csv' are in the 'data/' directory.")
        raise


def explore_data(ratings):

    print("\n" + "=" * 70)
    print("Exploratory Data Analysis")
    print("=" * 70)
    print("\nRating Statistics:")
    print(ratings['rating'].describe())
    rating_probs = ratings['rating'].value_counts(normalize=True).sort_index()
    print("\nRating Distribution:")
    print(rating_probs)
    return rating_probs


def create_user_item_matrix(ratings):

    user_item_matrix = ratings.pivot_table(
        index='userId', columns='movieId', values='rating'
    )
    sparsity = (user_item_matrix.isna().sum().sum() /
                (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100
    print(f"\nUser-Item Matrix Shape: {user_item_matrix.shape}")
    print(f"Matrix Sparsity: {sparsity:.2f}%")


    user_item_matrix_filled = user_item_matrix.fillna(0)
    return user_item_matrix_filled, user_item_matrix