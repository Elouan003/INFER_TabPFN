import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib 
import time 
import random 
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from pathlib import Path
from sklearn.neighbors import BallTree
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
""" Progress bar for fun """
# Source - https://stackoverflow.com/a/37630397
# Posted by Aravind Voggu, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-09, License - CC BY-SA 4.0

def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)






"""Importing data and defining paths"""
BASE_DIR = Path(__file__).resolve().parent.parent 

ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data/ExoMDN_data_2023.csv")




M_R_df = ExoMDN_data.iloc[:,[0,2]]# for now only mass and radius but need to consider interoir temperatur too 
print(M_R_df.columns)
M_R_array = M_R_df.to_numpy()
print(M_R_array.shape)



#This function returns the  indexes of the n closest neighbours of the given planet in the parameter space (R,M,T_eq) 
# and the distance to these neighbours in the param space 
t10 =   time.clock_gettime(0)
Tree = BallTree(M_R_array, leaf_size=40)
t11 =   time.clock_gettime(0)

print("timt to build tree : ",t11-t10, " seconds ")
def Neighbours(planet, tree, n=500):

    planet = np.atleast_2d(planet)

    dist, ind = Tree.query(planet, k=n)

    return dist, ind




""" First test of the Neighbours function """

"""
planet = M_R_array[150]
neighbour_distances,neighbour_indices = Neighbours(planet,M_R_array)


#sample data for quicker plot 
sample = M_R_array[::10]

# remove self
neighbour_indices = neighbour_indices[0][1:]
neighbours = M_R_array[neighbour_indices]
print(neighbours)
# plot
ax = plt.figure(figsize=(8,6))

# background points
plt.scatter(sample[:,1], sample[:,0], s=1, alpha=0.2, label="Dataset (1/10)")

# nearest neighbours
plt.scatter(neighbours[:,1], neighbours[:,0],
            color="red", s=40, marker = ".", label="Nearest neighbours")

# query planet
plt.scatter(planet[1], planet[0],
            color="black", s=80, marker="x", label="Query planet")

plt.xlabel("Mass")
plt.ylabel("Radius")
plt.legend()
plt.title("Nearest neighbours in Mass–Radius space")
plt.xlim = (10,15)
plt.ylim = (1.5,2.5)

plt.show()
"""


""" We now test how much precision we gain and how much time it costs to make predictions with the 500 points selected in different ways such as 
random in the whole set, radnom in a sphere, or 500 closest neighbors  """

"""Random training """

#select testing sample that is the same for both methods 
train_indices = random.choices(range(0,ExoMDN_data.shape[0]),k=100)

radius_test = ExoMDN_data.iloc[train_indices,2]
interior_test = ExoMDN_data.iloc[train_indices,3:17]
# First run as usual with 500 random points for the training set
predictions = []
prediction_time  = []
initialize_time = []
print("Computing predictions with random training set ")
for i in range (100) : 
    
    t1 = time.clock_gettime(0)
    random_indices = random.choices(range(0,ExoMDN_data.shape[0]),k=500)
    random_interior_train = ExoMDN_data.iloc[random_indices,3:17]
    random_radius_train = ExoMDN_data.iloc[random_indices,2]

    reg = TabPFNRegressor()
    reg.fit(random_interior_train,random_radius_train)

    t2 = time.clock_gettime(0)
    prediction = reg.predict(interior_test.iloc[[i]])
    t3 = time.clock_gettime(0)
    predictions.append(prediction)
    initialize_time.append(t2-t1)
    prediction_time.append(t3-t2)
    progress_bar(i,100)

            



print("________Results for traning on random 500 planets ________")
print ("Time for 100 predictions : " ,np.sum(prediction_time)," or for each prediction :", np.average(prediction_time))
print ("Time for 100 initializations(random) : " ,np.sum(initialize_time)," or for each initialization  :", np.average(initialize_time))
print("The average MSE over the 100 planets is : ", mean_squared_error(radius_test, predictions))
print("The R^2 score over the 100 planets is : ", r2_score(radius_test, predictions))

"""Closest neigbours"""
# we use the same testing indices and testing sets for the resluts to be comparable and only change on what data we "train"  the model ! 
predictions = []
prediction_time  = []
initialize_time = []
print("Computing predictions with closest neighbours ")
for i in range (100) : 
    
    t1 = time.clock_gettime(0)
    neighbour_distances,neighbour_indices = Neighbours(ExoMDN_data.iloc[i,[0,2]],M_R_array)
    neighbour_interior_train = ExoMDN_data.iloc[neighbour_indices[0],3:17]
    neighbour_radius_train = ExoMDN_data.iloc[neighbour_indices[0],2]
   
    reg = TabPFNRegressor()
    reg.fit(neighbour_interior_train,neighbour_radius_train)
    t2 = time.clock_gettime(0)
    prediction = reg.predict(interior_test.iloc[[i]])
    t3 = time.clock_gettime(0)
    predictions.append(prediction)
    initialize_time.append(t2-t1)
    prediction_time.append(t3-t2)
    progress_bar(i,100)

print("________Results for traning on 500 closest neigbours from parameter space ________")
print ("Time for 100 predictions : " ,np.sum(prediction_time)," or for each prediction :", np.average(prediction_time))
print ("Time for 100 initializations(random) : " ,np.sum(initialize_time)," or for each initialization  :", np.average(initialize_time))
print("The average MSE over the 100 planets is : ", mean_squared_error(radius_test, predictions))
print("The R^2 score over the 100 planets is : ", r2_score(radius_test, predictions))
    
"Random selection in a given subset of parameter space"""

def random_in_subset(planet, tree, n=500): 
    planet = np.atleast_2d(planet)

    dist, ind = Tree.query(planet, k=n)

    return dist, ind


