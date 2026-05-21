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


"""Data selection"""
#ExoMDN_data = ExoMDN_data[ExoMDN_data["planet_radius"]<4]
ExoMDN_data["atmosphere_mf_log"] = np.log10(ExoMDN_data["atmosphere_mf"])
#ExoMDN_data = ExoMDN_data[ExoMDN_data["atmosphere_mf_log"]>-6]
#ExoMDN_data = ExoMDN_data[ExoMDN_data["atmosphere_mf_log"]<-5]







radius_data = ExoMDN_data["planet_radius"]
interior_data = ExoMDN_data[["atmosphere_mf_log","water_mf","mantle_mf","core_mf","T_eq"]]
atm_mf_log = ExoMDN_data["atmosphere_mf_log"]


X = interior_data.values 
Y = radius_data.values 
Z = atm_mf_log.values

"""Split data into training,validation and testing"""

X_train, X_temp, y_train, y_temp,z_train,z_temp = train_test_split(X, Y, Z, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test,z_val,z_test   = train_test_split(X_temp, y_temp,z_temp, test_size=0.5, random_state=42)
# → ~5.4M training | ~300k validation | ~300k testing

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)


"""loading model """

model = xgb.Booster()
model.load_model(BASE_DIR/ "XG_boost_trained_gain_test.json")

"""predictions """
t0 = time.perf_counter()
y_pred = model.predict(dtest)
t1 = time.perf_counter()
print(f"Average prediction time = {(t1-t0)/len(y_test)} seconds ")
"""Precision diagnostics"""

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"  MAE  : {mae:.4f} R_earth")
print(f"  RMSE : {rmse:.4f} R_earth")
print(f"  R²   : {r2:.6f}")

""""Relative error diagnostic """
rel_error = np.abs(y_pred-y_test)/y_test
print("relative error XG_boost ", np.mean(rel_error))


rel_error_clean = rel_error[rel_error<0.5]
print("relative error XG_boost without tail ", np.mean(rel_error_clean))

""" R_R plotting """

plt.figure(figsize=(10, 9))
ax = plt.gca()
ax.plot(y_test,y_test,color = "plum",label = "1:1 line")
scatter = ax.scatter(y_test,y_pred,marker = ".",s=2,c =z_test ,cmap = "viridis",alpha=0.5)
plt.title("XG_Boost predicted planet_radius vs test values", fontsize=16)
plt.colorbar(scatter, ax=ax, label="atmosphere mf")
plt.ylabel("planet_radius predicted by XGBoost in given units", fontsize=16)
plt.xlabel("Validation planet_radius in given units", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(BASE_DIR / "data/plots/xg_boost_RR_plot_main.png")
plt.show()


"""Error histogramm plot"""
"""
plt.figure(figsize=(10, 9))
plt.title("Histogramm of XG_boost prediction error", fontsize=16)
plt.hist(y_test-y_pred, bins=100, alpha=0.7, label="error in R_earth",color = "teal")
plt.xlabel("XG_boost prediction error in R_earth", fontsize=16)
plt.savefig(BASE_DIR / "data/plots/xg_boost_error_hist.png")
plt.show()
"""

"""Speed testing"""
print(f"time for {len(y_test)} preds = {t1-t0} s  or {(t1-t0)/len(y_test)*1000} ms per prediction")


"MSE and precision "

print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R-squared (R^2):", r2_score(y_test, y_pred))




"""Error plotting on R_R diagramm"""
"""
plt.figure(figsize=(10, 9))
ax = plt.gca()
ax.plot(y_test,y_test,color = "plum",label = "1:1 line")
scatter = ax.scatter(y_test,y_pred,marker = ".",s=2,c =rel_error ,cmap = "inferno",alpha=0.5)
plt.title("XG_Boost predicted planet_radius vs test values", fontsize=16)
plt.colorbar(scatter, ax=ax, label="relative_error")
plt.ylabel("planet_radius predicted by XGBoost in given units", fontsize=16)
plt.xlabel("Validation planet_radius in given units", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(BASE_DIR / "data/plots/xg_boost_RR_plot_rel_error.png")
plt.show()


counts, bins = np.histogram(y_test,20)
plt.figure(figsize=(10, 8))  
plt.stairs(counts, bins)
plt.ylabel("Number of planets in data set", fontsize=12)
plt.xlabel("Radius of planet in Earth radii", fontsize=12)
plt.savefig(BASE_DIR  / "data/plots/xg_boost_subset_hist.png")
plt.show()



"relative error plot"
plt.figure(figsize=(10, 9))
ax = plt.gca()
scatter = ax.scatter(y_test,rel_error,marker = ".",alpha=0.5)
plt.title("XG_Boost predicted planet_radius vs test values", fontsize=16)
plt.ylabel("Relative error of the prediction", fontsize=16)
plt.xlabel("Validation planet_radius in given units", fontsize=16)
plt.axhline(y=np.mean(rel_error), color='r', linestyle='-',label="mean with tail")
plt.axhline(y=np.mean(rel_error_clean), color='b', linestyle='-',label="mean without tail" )

plt.legend(fontsize=16)
plt.savefig(BASE_DIR / "data/plots/xg_boost_rel_error_against_R.png")
plt.show()
"""
