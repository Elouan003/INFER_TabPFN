import numpy as np
import emcee
import matplotlib.pyplot as plt
import time
import pandas as pd 
from pathlib import Path
import random 
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
#Importing data from EXOMdn and setting paths 
BASE_DIR = Path(__file__).resolve().parent.parent 
ExoMDN_data = pd.read_csv(BASE_DIR  / "data/ExoMDN_data/ExoMDN_data_2023.csv")

# Here we make a very basic fitting of the model for a first test of the MCMC code 


random_indices = random.choices(range(0,ExoMDN_data.shape[0]),k=500) # Selects five hundreds indices at random in the data set 
random_interior_train = ExoMDN_data.iloc[random_indices,3:17]
random_radius_train = ExoMDN_data.iloc[random_indices,2]
reg = TabPFNRegressor()
reg.fit(random_interior_train,random_radius_train)



"""Likelihood function dervied similarly to Tijn et al. 2025 to be used in the MCMC evaluation later """
# prediction = reg.predict(interior_parameters)


def Likelihood(interior_parameters,observed_parameters= np.array([0,1,2])): # !! exterior parameters will be an input by the user once this is put into usable form 
    ### observed parameters are ###
    # planet_mass
    # T_eq 
    # planet_radius

    ### interior parameters has the components :###
    #atmosphere mf (mass fraction) 
    #atmosphere rf  (radius fraction)
    #water mf
    #ice mf
    #mantle mf 
    #mantle rf
    #core mf
    #core rf
    #log_d_mantle_core
    #log_m_mantle_core
    #log_d_atmosphere_core
    #log_m_atmosphere_core
    #log_d_water_core
    #log_m_water_core
    

    sigma = 0.025 # needs to be evaluatted from testing but leave like this for now 
    #observed_R = observed_parameters[2]
    predicted_R  = reg.predict(interior_parameters.reshape(1, -1))
    return  1  /   np.sqrt(2*np.pi*sigma**2) * np.e**(-0.5*  (predicted_R-observed_parameters[2])**2 / sigma**2  )


"""Running a first MCMC scheme with likelihood funtion above """

ndim = 14 # Dimension of interior parameters space
nwalkers = 40 # Number of walkers
p0 = 10* np.random.rand(nwalkers, ndim) # Initial grid
stretch_move = 2
start_time = time.time()
iterations = 100
burn_in = 10

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, Likelihood, moves=emcee.moves.StretchMove(a=stretch_move))

# Run mcmc algorithm for burn in 
state = sampler.run_mcmc(p0, burn_in) # What does MCMC do when I tell it to run the chains in parallel 
sampler.reset()

# Use state as new initial value 
sampler.run_mcmc(state, iterations)
samples = sampler.get_chain(flat=True)

end_time = time.time()

# retrieving the probabilities from the chains 
log_probe = sampler.get_log_prob(flat=True)

# retrieving the index of the sample with highest probabilty 
max_log_probe_index = np.argmax(log_probe)

# retrieving values of the parameters for the sample with highest probabilty 
max_likelihood = samples[max_log_probe_index]
time_mcmc = end_time - start_time

print("MCMC time = ", time_mcmc)
print("max likelihood ", max_likelihood)