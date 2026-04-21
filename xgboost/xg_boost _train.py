import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib 
import time 
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from pathlib import Path
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



"""Importing data from CSV, sorting and cleaning """

BASE_DIR = Path(__file__).resolve().parent.parent 

ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data_2023.csv")
print(ExoMDN_data.head())
print(ExoMDN_data.shape)
radius_data = ExoMDN_data["planet_radius"]
interior_data = ExoMDN_data.iloc[:,3:17]


print(interior_data.shape)
print(radius_data.shape)

X = interior_data.values 
Y = radius_data.values 

"""Split data into training,validation and testing"""

X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# → ~5.4M training | ~300k validation | ~300k testing

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    "objective":        "reg:squarederror",  # régression standard
    "eval_metric":      "rmse",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "tree_method":      "hist",   # rapide sur CPU
    "device":         "cuda",   # ← décommenter si GPU dispo
    "seed":             42,
}

evals = [(dtrain, "train"), (dval, "eval")]

class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training", unit="tree")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        # Affiche le dernier score de validation
        if evals_log:
            val_rmse = list(evals_log.get("eval", {}).get("rmse", [0]))[-1]
            self.pbar.set_postfix({"val_rmse": f"{val_rmse:.4f}"})
        return False  # False = continuer l'entraînement

    def after_training(self, model):
        self.pbar.close()
        return model

model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, "train"), (dval, "eval")],
    early_stopping_rounds=50,
    verbose_eval=False,  # désactiver le verbose natif
    callbacks=[TqdmCallback(total=2000)],
)
# Prédictions sur le test set
y_pred = model.predict(dtest)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f} | R²: {r2:.4f}")

# Feature importance (utile pour valider la physique)
xgb.plot_importance(model, importance_type="gain")

model.save_model("XG_boost_trained.json")