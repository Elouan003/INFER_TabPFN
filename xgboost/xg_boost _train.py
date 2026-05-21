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

#remove bigger radii and see what effect it has on our predictions for small planets 
#ExoMDN_data = ExoMDN_data[ExoMDN_data["planet_radius"]<4]
ExoMDN_data["atmosphere_mf_log"] = np.log10(ExoMDN_data["atmosphere_mf"])
#ExoMDN_data = ExoMDN_data[ExoMDN_data["atmosphere_mf_log"]>-6]
#ExoMDN_data = ExoMDN_data[ExoMDN_data["atmosphere_mf_log"]<-5]
#____________________________________________________________________________________________________________________________________
plt.figure(figsize=(10, 9))
plt.title("hist of atm mf log ", fontsize=16)
plt.hist(ExoMDN_data["atmosphere_mf_log"], bins=100, alpha=0.7, label="error in R_earth",color = "teal")
plt.xlabel("atm mf log ", fontsize=16)
plt.savefig(BASE_DIR / "data/plots/xg_boost_error_hist.png")
plt.show()
radius_data = ExoMDN_data["planet_radius"]
interior_data = ExoMDN_data[["atmosphere_mf_log","water_mf","mantle_mf","core_mf","T_eq"]]

print(interior_data.head())
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
    "objective":        "reg:squarederror",  # Standard regression
    "eval_metric":      "rmse",
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.5,
    "colsample_bylevel": 0.8,  # 80% des features par niveau
    "colsample_bynode": 0.8,   # 80% des features par noeud
    "min_child_weight": 3,
    "tree_method":      "hist",   # rapide sur CPU
    "device":         "cuda",   # for GPU use 
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
    early_stopping_rounds=100,
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

model.save_model("XG_boost_trained_gain_test.json")