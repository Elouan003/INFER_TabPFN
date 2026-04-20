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


from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor


"""Importing data from CSV, sorting and cleaning """


BASE_DIR = Path(__file__).resolve().parent.parent 

#ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data/ExoMDN_data_2023.csv")
#print(ExoMDN_data.columns)

#print(ExoMDN_data.shape)
#print(interior_data.shape)



"""  ___________________________Testing how much the precision scales with the size of the sample used for the first fit_______________________"""
"""fitting_time = []
testing_time = []
R2_array = []
MSE_array = []

for i in range (1,300) :
    if (i%10 == 0 ): 
          print("Proress ", i/300*100, "%")
    #Building training sets of increasing size from 10 to 3000 in steps of 10 
    training_size = i*10
    #interior_data = ExoMDN_data.iloc[0:training_size,3:17]
    #R_data  = ExoMDN_data.iloc[0:3000,2]

    interior_train = ExoMDN_data.iloc[0:training_size,3:17]
    interior_test = ExoMDN_data.iloc[training_size:training_size+200,3:17]
    radius_train = ExoMDN_data.iloc[0:training_size,2]
    radius_test = ExoMDN_data.iloc[training_size:training_size+200,2]


    #Intialize the TabPFN regressor and running the predictions for the given training size 
    t1 = time.clock_gettime(0)
    reg = TabPFNRegressor()
    reg.fit(interior_train,radius_train)
    t2 = time.clock_gettime(0)
    
    #important to do this in batches not to overload memory !! If we do a conatainer to be run on other machines maybe this needs to be taken into account ...
    batch_size = 200   # try 100–500 depending on memory
    predictions = []

    for i in range(0, len(interior_test), batch_size):
            batch = interior_test[i:i + batch_size]
            preds = reg.predict(batch)
            predictions.append(preds)
        

    predictions = np.concatenate(predictions)
    t3 = time.clock_gettime(0)
    fitting_time.append(t2-t1)
    testing_time.append(t3-t2)
    R2_array.append(r2_score(radius_test, predictions))
    MSE_array.append(mean_squared_error(radius_test, predictions))
    
  









#Plotting the different diagnostics for each training samople size 
np.savez("arrays.npz", a=fitting_time, b=testing_time, c=R2_array,d=MSE_array)# done and saved to arrays.npz
"""
data = np.load("arrays.npz")

fitting_time = data["a"]
testing_time = data["b"]
R2_array= data["c"]
MSE_array = data["d"]

training_size_array = np.linspace(1,300,299)*10


plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(training_size_array,fitting_time,marker = ".", color = "teal")
plt.title(f"Fitting time for given fitting sample size ")
plt.ylabel("Fitiing time [s]", fontsize=12)
plt.xlabel("Size of training sample", fontsize=12)
plt.savefig( BASE_DIR / "data/Plots/Fitting_time.png")
plt.show()


plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(training_size_array,testing_time,marker = ".", color = "teal")
plt.title(f"Prediction time for given fiting sample size")
plt.ylabel("prediction time for one point [s]", fontsize=12)
plt.xlabel("Size of training sample", fontsize=12)
plt.savefig(BASE_DIR / "data/Plots/Prediction_time.png")
plt.show()


plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(training_size_array,MSE_array,marker = ".", color = "teal")
plt.title(f"MSE for given fitting sample size")
plt.ylabel("Mean squared error [R_Earth]", fontsize=12)
plt.xlabel("Size of training sample", fontsize=12)
plt.savefig(BASE_DIR  / "data/Plots/MSE.png")
plt.show()


plt.figure(figsize=(10, 8))
ax = plt.gca()
ax.scatter(training_size_array,R2_array,marker = ".", color = "teal")
plt.title(f"R^2 for given training sample size")
plt.ylabel("R squared", fontsize=12)
plt.xlabel("Size of training sample", fontsize=12)
plt.savefig(BASE_DIR  / "data/Plots/R2.png")

plt.show()



