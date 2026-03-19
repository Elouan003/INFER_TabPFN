import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib 
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from pathlib import Path
from progress.bar import Bar
import time 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor


"""Importing data from CSV, sorting and cleaning """


BASE_DIR = Path(__file__).resolve().parent
ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data/ExoMDN_data_2023.csv")
print(ExoMDN_data.columns)
interior_data = ExoMDN_data.iloc[0:3000,3:17]
print(ExoMDN_data.shape)
print(interior_data.shape)

#Here we split our data into a training and testing part and set a random state for reproducibility 
"""interior_train, interior_test, radius_train, radius_test = train_test_split(
    interior_data,
    ExoMDN_data.iloc[0:3000,2],
    test_size=  0.1,
    random_state= 42,
)"""
interior_test = ExoMDN_data.iloc[3000:4000,3:17]
radius_test = ExoMDN_data.iloc[3000:4000,2]

#Intialize the TabPFN regressor 

t1= time.clock_gettime(0)

reg = joblib.load(BASE_DIR /"tabpfn_save.pkl")


#Make predictions about the observables with PFN 
#important to do this in batches not to overload memory !! If we do a conatainer to be run on other machines maybe this needs to be taken into account ...
t2 = time.clock_gettime(0)

batch_size = 200   # try 100–500 depending on memory
predictions = []

for i in range(0, len(interior_test), batch_size):
        batch = interior_test[i:i + batch_size]
        preds = reg.predict(batch)
        predictions.append(preds)
    

predictions = np.concatenate(predictions)
t3 = time.clock_gettime(0)

#some time diagnostics on the predictions 
print("Run_time for the initialization  from file : ",t2-t1)
print("Run_time for the predictions : ",t3-t2 ," seconds in totoal or ", (t3-t2)/1000, " seconds for a single prediction")



plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(radius_test,predictions,marker = ".", color = "teal")
plt.title(f"Training sample size = 3000 test sample size = {radius_test.size}")
plt.ylabel("Predicted radius in Earth radii", fontsize=12)
plt.xlabel("Radius from ExoMDN in Earth radii", fontsize=12)
plt.plot(radius_test,radius_test,color = "orange")
plt.show()

# Absolute errors 

Errors = np.abs(predictions - radius_test)

plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(radius_test,Errors,marker = ".", color = "teal")
plt.title(f"Training sample size = 3000, test sample size = {radius_test.size}")
plt.ylabel("absolute error", fontsize=12)
plt.xlabel("Radius from ExoMDN in Earth radii", fontsize=12)
plt.show()


#Error diagnostics 
print("Mean Squared Error (MSE):", mean_squared_error(radius_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(radius_test, predictions))
print("R-squared (R^2):", r2_score(radius_test, predictions))

