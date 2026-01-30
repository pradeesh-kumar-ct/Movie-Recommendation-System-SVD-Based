

import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate_model(predictions, user_item_matrix):

    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)

    actual = user_item_matrix.values

    mask = actual > 0

    actual_ratings = actual[mask]
    predicted_ratings = predictions[mask]

    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = np.mean(np.abs(actual_ratings - predicted_ratings))

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print("=" * 70 + "\n")

    return rmse, mae