#file -- synth_fns.py

## functions for inference example with synthetic data

import numpy as np
import os
import pickle
import pprint
import time

import pyross

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_model(popN) :
    gamma = 1
    death_ratio = 0.02

    parameters_true = {
        'beta' : 0.02*7,
        'gE' : 0.2*7,
        'gR' : gamma*(1-death_ratio),
        'gD' : gamma*death_ratio
    }

    model_spec = {
        "classes" : ["S", "E", "I", "D"],

        "S" : {
            "infection" : [ ["I", "-beta"] ]
        },
        "E" : {
            "infection" : [ ["I", "beta"] ],
            "linear"    : [ ["E", "-gE"] ]
        },
        "I" : {
            "linear"    : [ ["I", "-gR"] , ["I", "-gD"], ["E", "gE"] ]
        },
        "D" :{
            "linear"    : [ ["I", "gD"] ]
        }
    }

    M = 1                # one age cohort
    Ni = np.array([popN])   # cohort population list

    # set the contact structure
    C = np.array([[20.]])


    def contactMatrix(t):
        return C

    return { 'mod' : model_spec ,
           'CM' : contactMatrix ,
           'params' : parameters_true ,
           'cohortsM' : M ,
           'cohortsPop' : Ni ,
           'popN' : popN ,
         }

# leapEps <= 0 (or default) is Gillespie method
# leapEps > 0 is tau-leaping with tolerance parameter epsilon
def make_stochastic_traj(Tf, Nf, random_seed, model_dict, leapEps=-1):

    ## add seed to model struct
    model_dict['params']['seed'] = random_seed

    ## relabel parameters, this code was borrowed
    N = model_dict['popN']
    model_spec = model_dict['mod']
    contactMatrix = model_dict['CM']
    parameters_true = model_dict['params']
    M = model_dict['cohortsM']
    Ni = model_dict['cohortsPop']
    
    # (RLJ modified to use fixed fraction of N)
    I0 = 40 * N/1e5
    R0  = 0
    D0 = 0
    E0 = 100 * N/1e5
    S0  = N - I0 - E0 - R0 - D0

    x0_true = np.array([S0, E0, I0, D0], dtype='float')
    
    # use pyross stochastic to generate traj and save
    print(parameters_true)
    sto_model = pyross.stochastic.Spp(model_spec, parameters_true, M, Ni)
    if leapEps <= 0 :  ## gillespie
        data = sto_model.simulate(x0_true, contactMatrix, Tf, Nf)
    else :
        data = sto_model.simulate(x0_true, contactMatrix, Tf, Nf, method='tau_leaping', epsilon=leapEps)

    data_array = data['X']

    #print(Tf,Nf)
    #print(data['t'])
    
    return data_array

def get_start_time(data_array, popN, fracDeaths):
    # get the start time for when deaths first hit nDeaths
    deaths = data_array[:, -1]
    Tf_start = min(np.argwhere(deaths >= popN*fracDeaths)[:, 0])
    return Tf_start  ## note this is an integer index


def get_estimator(highAccuracy,model_dict, data_array, N, Nf_start, Nf_inference) :
    ## relabel parameters, this code was borrowed
    N = model_dict['popN']
    model_spec = model_dict['mod']
    contactMatrix = model_dict['CM']
    parameters_true = model_dict['params']
    M = model_dict['cohortsM']
    Ni = model_dict['cohortsPop']

    x = (data_array[Nf_start:Nf_start+Nf_inference]).astype('float')

    ## filter for partially observed data
    fltr = np.array([[0, 0, 0, 1]], dtype='float')
    ## the observed data
    obs=np.einsum('ij,kj->ki', fltr, x)
    ## the true initial condition (at start of inference period)
    trueInit=x[0]

    estimator = pyross.inference.Spp(model_spec, parameters_true, M, Ni, Omega=1, lyapunov_method='euler')
    #estimator.set_lyapunov_method('euler')

    ## we require very high accuracy which is achieved by these settings
    if highAccuracy:
        print('setting high-accuracy for likelihood')
        estimator.set_det_method('LSODA',rtol=1e-9)
        estimator.set_lyapunov_method('LSODA',rtol=1e-9)

    #estimator.set_det_method('LSODA',rtol=1e-6)
    #estimator.set_lyapunov_method('LSODA',rtol=1e-6)

    return [estimator,fltr,obs,trueInit]

def get_priors(model_dict,betaOffset,betaPriorLogNorm,fracDeaths,estimator):
    ## relabel parameters, this code was borrowed
    N = model_dict['popN']
    model_spec = model_dict['mod']
    contactMatrix = model_dict['CM']
    parameters_true = model_dict['params']
    M = model_dict['cohortsM']
    Ni = model_dict['cohortsPop']

    priorMeanBeta = betaOffset*model_dict['params']['beta']

    # make parameter guesses and set up bounds for each parameter
    eps=1e-4
    param_priors = {
        'beta':{
            'mean': priorMeanBeta,
            'std': priorMeanBeta/2,
            'bounds': [eps, priorMeanBeta*5 ]
        },
    }
    if not betaPriorLogNorm : 
      param_priors['beta']['prior_fun'] = 'truncnorm'
    
    v = estimator.find_fastest_growing_lin_mode(0)
    v = v*fracDeaths*N/v[-1] # scale lin mode to fit the number of deaths
    
    E0_g = v[1]
    E0_std = E0_g/3
    E0_bounds = [E0_g*0.1, E0_g*10]

    S0_g = N + v[0]
    S0_std = E0_std
    S0_bounds = [S0_g/10, N]

    I0_g = v[2]
    I0_std = I0_g/3
    I0_bounds = [I0_g*0.1, I0_g*10]

    init_priors = {
        'independent':{
            'fltr':np.array([True, True, True, False]),
            'mean': np.array([S0_g, E0_g, I0_g]),
            'std': np.array([S0_std, E0_std, I0_std]),
            'bounds': [S0_bounds, E0_bounds, I0_bounds],
            'prior_fun': 'truncnorm'
        }
    }
    #print('initial condition guess:', S0_g, E0_g, I0_g)

    return [param_priors,init_priors]


def do_inf(estimator, obs, fltr, data_array,
           N, Tf_inference, random_seed, param_priors, init_priors, model_dict, atol) :

    ## relabel parameters, this code was borrowed
    #N = model_dict['popN']
    #model_spec = model_dict['mod']
    contactMatrix = model_dict['CM']
    #parameters_true = model_dict['params']
    #M = model_dict['cohortsM']
    #Ni = model_dict['cohortsPop']
    
    ftol=1e-5  ## this is for local optimiser (relative tol)

    return estimator.latent_infer(obs, fltr, Tf_inference, param_priors, init_priors,
                                        contactMatrix=contactMatrix, enable_global=True,
                                        global_max_iter=1000, global_atol=atol, local_max_iter=10000,
                                        cma_population=64, cma_random_seed=random_seed,
                                        verbose=True, ftol=ftol)

def sliceLikelihood(rangeParam,infResult,estimator,obsData,fltrDeath,contactMatrix,Tf_inference) :
    nMrange = 101
    mFactors = np.linspace(1-rangeParam,1+rangeParam,nMrange)

    paramsCopy = infResult['params_dict'].copy()

    ## just check we copied successfully(!)
    paramsCopy['beta'] *= 0.9
    #pprint.pprint(infResult['params_dict'])
    #pprint.pprint(paramsCopy)

    likVals = []
    for mm in mFactors :
        paramsCopy['beta'] = mm * infResult['params_dict']['beta']
        logpVal = -estimator.minus_logp_red(paramsCopy, infResult['x0'], obsData, fltrDeath,
                                            Tf_inference, contactMatrix, tangent=False)
        likVals = likVals + [logpVal]

    return [mFactors*infResult['params_dict']['beta'],likVals]



def do_mcmc(nSamples, nProcMCMC, estimator, 
            Tf_inference, infResult, obsDeath, fltrDeath,
            param_priors, initPriors, model_dict, seed=None) :
    contactMatrix = model_dict['CM']

    dim = np.size(infResult['flat_params'])
    print('est map',infResult['flat_params'],dim)

    initWalk= np.repeat([infResult['flat_params']],repeats=(dim*2),axis=0)
    #print('param shape',np.shape(initWalk))

    ## add some noise to the initial parameters, else the MCMC is not happy
    pertWalk = np.random.uniform( size=np.shape(initWalk) )
    pertSize = 1.0/20
    pertWalk = 1.0 + pertWalk*pertSize
    #print(pertWalk)

    initWalk *= pertWalk
    
    xArgs = {
        'nsamples' : nSamples,
        'walker_pos' : initWalk
    }
    if nProcMCMC != None :
        xArgs['nprocesses'] = nProcMCMC
   
    ## this can be used to ensure reproducibility
    if seed != None :
        np.random.seed(seed)
    sampler = estimator.latent_infer_mcmc(obsDeath, fltrDeath, Tf_inference,
                                    param_priors,
                                    initPriors, # initPriorsLinMode,
                                    contactMatrix = contactMatrix, # generator=contactBasis,
                                    #intervention_fun=interventionFn,
                                    tangent=False,
                                    verbose=True,
                                    **xArgs,
                                    #nsamples=runSampInit,
                                    #nprocesses=nProcMCMC,
                                    #walker_pos = initWalk
                                )
    return sampler


def load_mcmc_result(estimator, obsDeath, fltr,  sampler, param_priors,initPriors, model_dict):
    contactMatrix = model_dict['CM']

    pp = sampler.get_log_prob()
    nSampleTot = np.shape(pp)[0]
    
    result_mcmc = estimator.latent_infer_mcmc_process_result(sampler, obsDeath, fltr,
                                            param_priors,
                                            initPriors, # initPriorsLinMode,
                                            contactMatrix = contactMatrix, # generator=contactBasis,
                                            #intervention_fun=interventionFn,
                                            discard=int(nSampleTot/3),
                                            thin=int(nSampleTot/100),
                                            )
    return result_mcmc


