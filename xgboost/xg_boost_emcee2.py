import numpy as np
import emcee
import matplotlib.pyplot as plt
import time
import pandas as pd 
from pathlib import Path
import random 
import xgboost as xgb
from tqdm import tqdm
import corner 
from scipy.stats import dirichlet



BASE_DIR = Path(__file__).resolve().parent.parent 


def log_likelihood(interior_parameters,observed_parameters): # !! exterior parameters will be an input by the user once this is put into usable form 
    ### observed parameters are ###
    # planet_mass
    # T_eq 
    # planet_radius

    ### interior parameters has the components :###
    #atmosphere mf 
    #water mf
    #mantle mf 
    #core mf
  
    

    sigma = 0.2 # needs to be evaluatted from testing but leave like this for now 
    d_interior = xgb.DMatrix(interior_parameters.reshape(1, -1))   
    predicted_R  = model.predict(d_interior)[0]
    return -0.5 * ((predicted_R - observed_parameters[2]) / sigma)**2 #log probabilty as a gaussian around observed Radius 








def sample_valid_p0(nwalkers, ndim, alpha):
    p0 = []
    while len(p0) < nwalkers:
        # Sample from Dirichlet then add small jitter 
        candidate = np.random.dirichlet(alpha)
        candidate += 1e-3* np.abs(np.random.randn(ndim))  # jitter
        candidate = np.abs(candidate)
        if np.isfinite(log_prior(candidate)):
            p0.append(candidate)
    return np.array(p0)














ndim = 4 # Dimension of interior parameters space
nwalkers = 40 # Number of walkers
p0 =sample_valid_p0(nwalkers,4,alpha=[0.1,0.6,2,2]) # Prior on internal parameters using a dirichlet distribution with a very low mf for the atmosphere and ahiger one for the core/mantel 
iterations = 1000
burn_in = 200
print("p0 = ",p0)








# Initialize the sampler
sampler= emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, moves=[
    (emcee.moves.DEMove(), 0.8),
    (emcee.moves.DESnookerMove(), 0.2)
])
