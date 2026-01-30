

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent



DATA_DIR = Path("C:/Users/Kshore N/PycharmProjects/PythonProject2/Movie-Recommendation-System-SVD-Based") / "data"
RATINGS_FILE = DATA_DIR / "u.csv"
MOVIES_FILE = DATA_DIR / "mm.csv"
SVD_N_FACTORS = 50