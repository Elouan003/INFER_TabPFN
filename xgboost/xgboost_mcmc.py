import numpy as np
import emcee
import matplotlib.pyplot as plt
import time
import pandas as pd 
from pathlib import Path
import random 
import xgboost as xgb
from tqdm import tqdm




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

"loading model"
model = xgb.Booster()
model.load_model(BASE_DIR/ "XG_boost_trained.json")


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
    

    sigma = 0.155 # needs to be evaluatted from testing but leave like this for now 
    #observed_R = observed_parameters[2]
    d_interior= xgb.DMatrix(interior_parameters,  label=interior_parameters)
    predicted_R  = model.predict(d_interior)
    return  1  /   np.sqrt(2*np.pi*sigma**2) * np.e**(-0.5*  (predicted_R-observed_parameters[2])**2 / sigma**2  )




"""Running MCMC scheme with likelihood funtion above """

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