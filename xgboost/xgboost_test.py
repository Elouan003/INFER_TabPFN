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


"""loading model """

model = xgb.Booster()
model.load_model(BASE_DIR/ "XG_boost_trained.json")

"""predictions """
t0 = time.perf_counter()
y_pred = model.predict(dtest)
t1 = time.perf_counter()


""" plotting """

plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.plot(y_test,y_test,color = "plum",label = "1:1 line")
ax.scatter(y_test,y_pred,marker = ".", color = "teal")
plt.title("XG_Boost predicted planet_radius vs test values")
plt.ylabel("planet_radius predicted by XGBoost in given units", fontsize=12)
plt.xlabel("Validation planet_radius in given units", fontsize=12)
plt.legend()
plt.show()
plt.savefig(BASE_DIR / "data/plots/xg_boost_RR_plot")




"""Speed testing"""
print(f"time for {len(y_test)} preds = {t1-t0} s  or {(t1-t0)/len(y_test)*1000} ms per prediction")
