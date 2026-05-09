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
ExoMDN_sample = ExoMDN_data.sample(n=500_000, random_state=42)
radius_data = ExoMDN_data["planet_radius"]
interior_data = ExoMDN_data[["atmosphere_mf","water_mf","mantle_mf","core_mf"]]




X = interior_data.values 
Y = radius_data.values 

"""Split data into training,validation and testing"""

X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# → ~5.4M training | ~300k validation | ~300k testing




"""Load regression from saved file """
regr = joblib.load("randomtree_200k.joblib")


"""Run predictions"""
y_pred = regr.predict(X_test)

"""Precision diagnostics"""

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"  MAE  : {mae:.4f} R_earth")
print(f"  RMSE : {rmse:.4f} R_earth")
print(f"  R²   : {r2:.6f}")



""" plotting """

plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.plot(y_test,y_test,color = "plum",label = "1:1 line")
ax.scatter(y_test,y_pred,marker = ".", color = "teal",alpha=0.01)
plt.title("Random_tree predicted planet_radius vs test values")
plt.ylabel("planet_radius predicted by Random_tree in given units", fontsize=12)
plt.xlabel("Validation planet_radius in given units", fontsize=12)
plt.legend()
plt.savefig(BASE_DIR / "data/plots/Random_tree_RR_plot")
plt.show()
