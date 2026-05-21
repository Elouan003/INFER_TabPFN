import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import joblib 
import time 
import random 
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from pathlib import Path
from sklearn.neighbors import BallTree
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
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

ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data_2023.csv")
interior_data  =  ExoMDN_data[["atmosphere_mf","water_mf","mantle_mf","core_mf","T_eq"]]
radius_data = ExoMDN_data["planet_radius"]

"""Defining testing planets for all the different methods to make sure our results are consistent """
"""
test_indices = random.choices(range(0,ExoMDN_data.shape[0]),k=100) #selects a hundred planets from the data set to be used as validation throught the run 
radius_test = ExoMDN_data.iloc[test_indices,2]
planets_test = ExoMDN_data.iloc[test_indices,[0,2]]
interior_test = interior_data.iloc[test_indices]
print(radius_test.shape)
print(interior_test.shape)
interior_data.drop(test_indices, inplace=True)
ExoMDN_data.drop(test_indices, inplace=True)
"""
M_R_df = ExoMDN_data.iloc[:,[0,2]]
print(M_R_df.columns)
M_R_array = M_R_df.to_numpy()
print(M_R_array.shape)



#This function returns the  indexes of the n closest neighbours of the given planet in the parameter space (R,M,T_eq) 
# and the distance to these neighbours in the param space 
t10 =   time.perf_counter()
Tree = BallTree(M_R_array, leaf_size=40)
t11 =   time.perf_counter()

print("timt to build tree : ",t11-t10, " seconds ")
def Neighbours(planet, tree, n=500):

    planet = np.atleast_2d(planet)

    dist, ind = tree.query(planet, k=n) # return the distance to n closest neigbours as well as the index of these points 

    return dist, ind




""" First test of the Neighbours function """
"""

planet = M_R_array[150]
neighbour_distances,neighbour_indices = Neighbours(planet,Tree)


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
"""First test of the random in local space around point """

def random_in_subset(planet, tree, n=500): 
    planet = np.atleast_2d(planet)

    indices_of_points_closeby= tree.query_radius(planet,r=2)[0]
    points_closeby = M_R_array[indices_of_points_closeby]
    #what to do if there are not 500 planets closeby (our data point is an outlier ? )
    selection_indices = random.choices(range(0,points_closeby.shape[0]),k=500)# selecting which of the points inside the subset we pick 
    indices = indices_of_points_closeby[selection_indices]
    return  indices

"""
planet = M_R_array[150]
neighbour_indices = random_in_subset(planet,Tree)

#sample data for quicker plot 
sample = M_R_array[::10]
# remove self
neighbour_indices = neighbour_indices[1:]
neighbours = M_R_array[neighbour_indices]

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
plt.title("Random points in selected subsection of parameter space")
plt.xlim = (10,15)
plt.ylim = (1.5,2.5)

plt.show()
"""

"""plot of random points in parameter space for illustration purposes """
"""
planet = M_R_array[150]
neighbour_indices = random.choices(range(0,ExoMDN_data.shape[0]),k=500)

#sample data for quicker plot 
sample = M_R_array[::10]
# remove self
neighbour_indices = neighbour_indices[1:]
neighbours = M_R_array[neighbour_indices]

# plot
ax = plt.figure(figsize=(8,6))

# background points
plt.scatter(sample[:,1], sample[:,0], s=1, alpha=0.2, label="Dataset (1/10)")

# nearest neighbours
plt.scatter(neighbours[:,1], neighbours[:,0],
            color="red", s=40, marker = ".", label="Fitting planets")

# query planet
plt.scatter(planet[1], planet[0],
            color="black", s=80, marker="x", label="Query planet")

plt.xlabel("Mass")
plt.ylabel("Radius")
plt.legend()
plt.title("Random planets in subspace")
plt.xlim = (10,15)
plt.ylim = (1.5,2.5)

plt.show()





"""

""" We now test how much precision we gain and how much time it costs to make predictions with the 500 points selected in different ways such as 
random in the whole set, radnom in a sphere, or 500 closest neighbors  """

"""Random training """

#select testing sample that is the same for both methods 


"""
# First run as usual with 500 random points for the training set
predictions = []
prediction_time  = []
initialize_time = []
print("Computing predictions with random training set ")
for i in range (100) : 
    
    t1 = time.perf_counter()
    random_indices = random.choices(range(0,ExoMDN_data.shape[0]),k=500)
    random_interior_train = interior_data.iloc[random_indices]
    random_radius_train = ExoMDN_data.iloc[random_indices,2]

    reg = TabPFNRegressor()
    reg.fit(random_interior_train,random_radius_train) #only in one dimension for radius !! 


    t2 = time.perf_counter()
    prediction = reg.predict(interior_test.iloc[[i]])
    t3 = time.perf_counter()
    predictions.append(prediction)
    initialize_time.append(t2-t1)
    prediction_time.append(t3-t2)
    progress_bar(i,100)

            



print("________Results for fitting on random 500 planets ________")
print ("Time for 100 predictions : " ,np.sum(prediction_time)," or for each prediction :", np.average(prediction_time))
print ("Time for 100 initializations(random) : " ,np.sum(initialize_time)," or for each initialization  :", np.average(initialize_time))
print("The average MAE over the 100 planets is : ", mean_absolute_error(radius_test, predictions))
print("The average percentage error  over the 100 planets is : ", mean_absolute_percentage_error(radius_test, predictions))
print("The R^2 score over the 100 planets is : ", r2_score(radius_test, predictions))
"""

"""Closest neigbours"""

"""
# we use the same testing indices and testing sets for the resluts to be comparable and only change on what data we "train"  the model ! 
predictions = []
prediction_time  = []
initialize_time = []


print("Computing predictions with closest neighbours ")
for i in range (100) : 
    
    t1 = time.perf_counter()
    neighbour_distances,neighbour_indices = Neighbours(planets_test.iloc[i],Tree) 
    #can not use radius test here we need the planet argument to be 2 dim 
    #with Mass and Radius 
    neighbour_interior_train = interior_data.iloc[neighbour_indices[0]]
    neighbour_radius_train = ExoMDN_data.iloc[neighbour_indices[0],2]
   
    reg = TabPFNRegressor()
    reg.fit(neighbour_interior_train,neighbour_radius_train)
    t2 = time.perf_counter()
    prediction = reg.predict(interior_test.iloc[[i]])
    t3 = time.perf_counter()
    
    predictions.append(prediction)
    initialize_time.append(t2-t1)
    prediction_time.append(t3-t2)
    progress_bar(i,100)


print("________Results for traning on 500 closest neigbours from parameter space ________")
print ("Time for 100 predictions : " ,np.sum(prediction_time)," or for each prediction :", np.average(prediction_time))
print ("Time for 100 initializations(random) : " ,np.sum(initialize_time)," or for each initialization  :", np.average(initialize_time))
print("The average MAE over the 100 planets is : ", mean_absolute_error(radius_test, predictions))
print("The average percentage error  over the 100 planets is : ", mean_absolute_percentage_error(radius_test, predictions))

print("The R^2 score over the 100 planets is : ", r2_score(radius_test, predictions))
"""
"""Random selection in a given subset of parameter space"""
"""
print(planets_test.iloc[0])



# we use the same testing indices and testing sets for the resluts to be comparable and only change on what data we "train"  the model ! 
predictions = []
prediction_time  = []
initialize_time = []
print("Computing predictions with random points in close neighbourhood  ")
for i in range (100) : 
    
    t1 = time.perf_counter()
    neighbour_indices = random_in_subset(planets_test.iloc[i],Tree)

    neighbour_interior_train = interior_data.iloc[neighbour_indices]
    neighbour_radius_train = ExoMDN_data.iloc[neighbour_indices,2]
   
    reg = TabPFNRegressor()
    reg.fit(neighbour_interior_train,neighbour_radius_train)
    t2 = time.perf_counter()
    prediction = reg.predict(interior_test.iloc[[i]])
    t3 = time.perf_counter()
    predictions.append(prediction)
    initialize_time.append(t2-t1)
    prediction_time.append(t3-t2)
    progress_bar(i,100)

print("________Results for traning on 500 random points in subspace close to planet in M_R space  ________")
print ("Time for 100 predictions : " ,np.sum(prediction_time)," or for each prediction :", np.average(prediction_time))
print ("Time for 100 initializations(random) : " ,np.sum(initialize_time)," or for each initialization  :", np.average(initialize_time))
print("The averageMAE over the 100 planets is : ", mean_absolute_error(radius_test, predictions))
print("The average percentage error  over the 100 planets is : ", mean_absolute_percentage_error(radius_test, predictions))

print("The R^2 score over the 100 planets is : ", r2_score(radius_test, predictions))
"""

"""Produce an R_R plot for the random points in neighbrhood """

ExoMDN_data["atmosphere_mf_log"] = np.log10(ExoMDN_data["atmosphere_mf"])
atm_mf_log = ExoMDN_data["atmosphere_mf_log"]


X = interior_data.values 
Y = radius_data.values 
Z = M_R_df.values



X_train, X_temp, y_train, y_temp,z_train,z_temp = train_test_split(X, Y, Z, test_size=0.1, random_state=42)
X_val, X_test, y_val, y_test,z_val,z_test   = train_test_split(X_temp, y_temp,z_temp, test_size=0.01, random_state=42)
# → ~5.4M training | ~570k validation | ~30k testing




predictions = []
prediction_time  = []
print("Computing predictions with random points in close neighbourhood  ")
for i,interior in enumerate(X_test) : 
    
    t1 = time.perf_counter()
    M_R = Z[i]
    neighbour_indices = random_in_subset(Z[i],Tree)

    neighbour_interior_train = interior_data.iloc[neighbour_indices]
    neighbour_radius_train = ExoMDN_data.iloc[neighbour_indices,2]
    interior  =  interior.reshape(1,-1)
    reg = TabPFNRegressor()
    reg.fit(neighbour_interior_train,neighbour_radius_train)
    t2 = time.perf_counter()
    prediction = reg.predict(interior)
    t3 = time.perf_counter()
    predictions.append(prediction)

    prediction_time.append(t3-t2)
    progress_bar(i,len(X_test))


np.savez("Tab_PFN_pred001.npz", a = y_test,b = predictions)# Tab_PFN_pred.npz


data = np.load("Tab_PFN_pred001.npz")
y_test = data["a"]
predictions = data["b"][:,0]


rel_error = np.abs(predictions-y_test)/y_test
print("mean rel error",np.mean(rel_error))
plt.figure(figsize=(10, 9))
ax = plt.gca()
ax.plot(y_test,y_test,color = "plum",label = "1:1 line")
scatter = ax.scatter(y_test,predictions,marker = ".",s=2,c =rel_error ,cmap = "inferno",alpha=0.5)
plt.title("XG_Boost predicted planet_radius vs test values", fontsize=16)
plt.colorbar(scatter, ax=ax, label="relative_error")
plt.ylabel("planet_radius predicted by XGBoost in given units", fontsize=16)
plt.xlabel("Validation planet_radius in given units", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(BASE_DIR / "data/plots/tab_pfn_RR_plot_rel_error.png")
plt.show()




log_atm = np.log10(X_test[:,0])
plt.figure(figsize=(10, 9))
ax = plt.gca()
ax.plot(y_test,y_test,color = "plum",label = "1:1 line")
scatter = ax.scatter(y_test,predictions,marker = ".",s=2,c = log_atm,cmap = "viridis",alpha=0.5)
plt.title("XG_Boost predicted planet_radius vs test values", fontsize=16)
plt.colorbar(scatter, ax=ax, label="Atmosphere mf")
plt.ylabel("planet_radius predicted by XGBoost in given units", fontsize=16)
plt.xlabel("Validation planet_radius in given units", fontsize=16)
plt.legend(fontsize=16)
plt.savefig(BASE_DIR / "data/plots/tab_pfn_RR_full.png")
plt.show()

counts, bins = np.histogram(y_test,20)
plt.figure(figsize=(10, 8))  
plt.stairs(counts, bins)
plt.ylabel("Number of planets in data set", fontsize=12)
plt.xlabel("Radius of planet in Earth radii", fontsize=12)
plt.savefig(BASE_DIR  / "data/plots/tab_pfn_subset_hist.png")
plt.show()