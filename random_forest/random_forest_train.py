import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib 
import time 
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

"""Importing data from CSV, sorting and cleaning """

BASE_DIR = Path(__file__).resolve().parent.parent 

ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data_2023.csv")
ExoMDN_sample = ExoMDN_data.sample(n=200_000, random_state=42)
radius_data = ExoMDN_data["planet_radius"]
interior_data = ExoMDN_data[["atmosphere_mf","water_mf","mantle_mf","core_mf","T_eq"]]




X = interior_data.values 
Y = radius_data.values 

"""Split data into training,validation and testing"""

X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# → ~5.4M training | ~300k validation | ~300k testing



" Initialize regressor and fit it to the data"
regr = RandomForestRegressor(
    n_estimators=200,       # 200 arbres — bon compromis vitesse/stabilité
    max_depth=25,           # profondeur suffisante pour capturer la physique
    min_samples_leaf=10,    # évite l'overfitting sur outliers
    max_features=0.5,       # ~7 features par split, accélère l'entraînement
    n_jobs=-1,              # utilise tous les cœurs CPU
    random_state=42,
    verbose=1,              # affiche la progression
)


print("\nTraining...")
t0 = time.time()
regr.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"Training done in {elapsed/60:.1f} min")

MODEL_PATH = BASE_DIR  / "randomtree_200k.joblib"
joblib.dump(regr, MODEL_PATH, compress=3) 
print(f"\nModel saved → {MODEL_PATH}")


