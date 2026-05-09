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

"""Importing data from CSV, sorting and cleaning """

BASE_DIR = Path(__file__).resolve().parent.parent 



"loading model"
model = xgb.Booster()
model.load_model(BASE_DIR/ "XG_boost_trained_v1.json")

alpha = [0.1, 1.0, 3.0, 3.0]

def log_prior(params): 
    #imposes limits on the values of the interior parameters during the MCMC run 

    #if np.any(params < 0) or np.any(params > 1): # no mass fraction should be >1 or <0 
    #    return -np.inf
    #if abs(np.sum(params) - 1.0) > 0.1:  # mass fractions should add up to 1 
        #return -np.inf
    return 0



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
  
    

    sigma = 0.155 # needs to be evaluatted from testing but leave like this for now 
    d_interior = xgb.DMatrix(interior_parameters.reshape(1, -1))   
    predicted_R  = model.predict(d_interior)[0]
    return -0.5 * ((predicted_R - observed_parameters[2]) / sigma)**2 #log probabilty as a gaussian around observed Radius 

def log_probability(interior_parameters,observed_parameters= np.array([1.081,112.269,1.140466]) ):
    #combines the likelihood function with consideration of constraints/prior on the interior parameters 
    lp = log_prior(interior_parameters)       # constraints on params 
    if not np.isfinite(lp):
        return -np.inf
    return float(lp + log_likelihood(interior_parameters, observed_parameters)) # sum not product since we work with log probabilities 




"""Running MCMC scheme with likelihood funtion above """


def sample_valid_p0(nwalkers, ndim, alpha):
    p0 = []
    while len(p0) < nwalkers:
        # Sample from Dirichlet then add small jitter before renormalizing
        candidate = np.random.dirichlet(alpha)
        candidate += 1e-3* np.abs(np.random.randn(ndim))  # jitter
        candidate = np.abs(candidate)
        #candidate /= candidate.sum()  # renormalize to simplex
        if np.isfinite(log_prior(candidate)):
            p0.append(candidate)
    return np.array(p0)



ndim = 4 # Dimension of interior parameters space
nwalkers = 40 # Number of walkers
p0 =sample_valid_p0(nwalkers,4,alpha=[0.1,0.6,2,2]) # Prior on internal parameters using a dirichlet distribution with a very low mf for the atmosphere and ahiger one for the core/mantel 
stretch_move = 2
start_time = time.time()
iterations = 1000
burn_in = 200
print("p0 = ",p0)
# Initialize the sampler
sampler= emcee.EnsembleSampler(nwalkers, ndim, log_probability, moves=[
    (emcee.moves.DEMove(), 0.8),
    (emcee.moves.DESnookerMove(), 0.2)
])


# Tests on prior distribution of walkers 


log_probs = [log_probability(p) for p in p0]
print("interoiors : ",p0  )
print("log_probabilities:", log_probs)
for i in range (len(p0)):
    print(np.sum(p0[i]))


print("Sont-elles toutes identiques ?", len(set(map(float, log_probs))) == 1)




alpha_uniforme = [1.0, 1.0, 1.0, 1.0]
alpha_physique = [0.05, 1.0, 3.0, 3.0]

samples_u = np.random.dirichlet(alpha_uniforme, size=10000)
samples_p = np.random.dirichlet(alpha_physique, size=10000)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, (name, s) in enumerate([("Uniforme", samples_u), ("Physique", samples_p)]):
    axes[i].hist(s[:, 0], bins=100, alpha=0.7, label="atmosphere_mf")
    axes[i].hist(s[:, 1], bins=100, alpha=0.7, label="water_mf")
    axes[i].hist(s[:, 2], bins=100, alpha=0.7, label="mantle_mf")
    axes[i].hist(s[:, 3], bins=100, alpha=0.7, label="core_mf")
    axes[i].set_title(f"Prior {name}")
    axes[i].legend()
    print(np.median(s[:, 0]))
#plt.show()







# Run mcmc algorithm for burn in 
state = sampler.run_mcmc(p0, burn_in,progress=True)

best = state.coords[np.argmax(state.log_prob)]




def samples_from_best(best, nwalkers, ndim, scale=1e-3):
    p0 = []
    while len(p0) < nwalkers:
        candidate = best + scale * np.random.randn(ndim)
        candidate = np.abs(candidate)
        if log_prior(candidate) > -np.inf and np.abs(np.sum(candidate) - 1.0)<= 0.1:
            p0.append(candidate)
        
    return np.array(p0)






print("log_prob:", state.log_prob)
print("coords:", state.coords)


sampler.reset()

sampler.run_mcmc(samples_from_best(best,nwalkers,4),iterations,progress=True)
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


"""Corner Plot"""

feature_names = ["atm_mf", "water_mf", "mantle_mf",
                 "core_mf"]
fig = corner.corner(samples, labels=feature_names, show_titles=True)
plt.savefig(BASE_DIR / "corner_4_features.pdf")
plt.show()