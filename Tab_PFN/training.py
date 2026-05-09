import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib 
import time 
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from pathlib import Path
from progress.bar import Bar

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor


"""Importing data from CSV, sorting and cleaning """


BASE_DIR = Path(__file__).resolve().parent.parent 
ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data_2023.csv")
print(ExoMDN_data.columns)
interior_data = ExoMDN_data[["atmosphere_mf","water_mf","mantle_mf","core_mf"]]
print(ExoMDN_data.shape)
print(interior_data.shape)

#Here we split our data into a training and testing part and set a random state for reproducibility 
"""
interior_train, interior_test, radius_train, radius_test = train_test_split(
    interior_data,
    ExoMDN_data.iloc[0:3000,2],
    test_size=  0.1,
    random_state= 42,
)""" 

interior_train = interior_data.iloc[0:3500]
interior_test = interior_data.iloc[3000:3500]
obs_train= ExoMDN_data.iloc[3000:3500,0:2]
obs_test =ExoMDN_data.iloc[3000:3500,0:2]
radius_train = ExoMDN_data.iloc[0:3500,2]
radius_test = ExoMDN_data.iloc[3000:3500,2]

#Intialize the TabPFN regressor and save the regressor to file 
t1 =  time.perf_counter()
reg = TabPFNRegressor()
reg.fit(interior_train,radius_train)
t2 = time.perf_counter()
joblib.dump(reg, "tabpfn_save.pkl")
t3 =  time.perf_counter()


#Make predictions about the observables with PFN 
#important to do this in batches not to overload memory !! If we do a conatainer to be run on other machines maybe this needs to be taken into account ...
batch_size = 200   # try 100–500 depending on memory
predictions = []

for i in range(0, len(obs_test), batch_size):
        batch = interior_test[i:i + batch_size]
        preds = reg.predict(batch)
        predictions.append(preds)
    

predictions = np.concatenate(predictions)
t4 =  time.perf_counter()


print("Run_time for the initialization of the regressor : ",t2-t1)
print("Run_time for the predictions : ",t4-t3 ," seconds in totoal or ", (t4-t3)/1000, " seconds for a single prediction")




"""
plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(radius_test,predictions,marker = ".", color = "teal")
plt.title(f"Training sample size = {radius_train.size}, test sample size = {radius_test.size}")
plt.ylabel("Predicted radius in Earth radii", fontsize=12)
plt.xlabel("Radius from ExoMDN in Earth radii", fontsize=12)
plt.plot(radius_test,radius_test,color = "orange")
plt.show()
"""""
"""
# Relative errors 

Errors = predictions - radius_test 

plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(radius_test,Errors,marker = ".", color = "teal")
plt.title(f"Training sample size = {radius_train.size}, test sample size = {radius_test.size}")
plt.ylabel("absolute error", fontsize=12)
plt.xlabel("Radius from ExoMDN in Earth radii", fontsize=12)
plt.show()


#Error diagnostics 
print("Mean Squared Error (MSE):", mean_squared_error(radius_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(radius_test, predictions))
print("R-squared (R^2):", r2_score(radius_test, predictions))

#Some diagnostics for the whole data set

counts, bins = np.histogram(ExoMDN_data["planet_radius"],100)
plt.figure(figsize=(10, 8))  
plt.stairs(counts, bins)
plt.ylabel("Number of planets in data set", fontsize=12)
plt.xlabel("Radius of planet in Earth radii", fontsize=12)
plt.show()
"""

#Error diagnostics 
print("Mean Squared Error (MSE):", mean_squared_error(radius_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(radius_test, predictions))
print("R-squared (R^2):", r2_score(radius_test, predictions))