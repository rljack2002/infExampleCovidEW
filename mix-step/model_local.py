#file -- model_local.py

## setup eng-wal model (independent of variant)

import numpy as np
from matplotlib import pyplot as plt
import pyross
import time 
import pandas as pd
import matplotlib.image as mpimg
import pickle
import os
import pprint

import scipy.stats

import expt_params_local
from ew_fns import *

def loadModel(exptParams,daysPerWeek,verboseMod,yesComet=False) :  ## yesComet does nothing (legacy)

    ## population and cohorts
    
    numCohorts = exptParams['numCohorts']  ## (to save typing, this is what pyross calls M)

    [fi, N, Ni, popDataAll] = readPopData(exptParams['popFile'],exptParams['numCohortsPopData'],numCohorts)

    if verboseMod :
        yMax = 1.2*np.max(fi)
        plt.ylim(0,yMax)
        plt.plot(fi,'o-')
        plt.show() ; plt.close()
        print('** totN',N)
        
        print('** population')
        print(popDataAll)

    # ### model and epidemiological params

    model_spec = getModThreeStage()
    print('** model\n')
    pprint.pprint(model_spec)

    modParams = getParamsThreeStage(numCohorts,popDataAll,daysPerWeek,verbose=verboseMod)

    if verboseMod :
        print('**inf params (with prior vals)')
        pprint.pprint(modParams)

    ## these are the "pyross rate parameters" as derived from the "model parameters"
    rateParams = paramMap(modParams)

    if verboseMod :
        print('\n **rate params')
        pprint.pprint(rateParams)

    ## contact matrix
    print("")
    contactBasis = setupCM(exptParams['chooseCM'],numCohorts,fi,verbose=verboseMod)

    ### data and basic inference settings

    deathCumulativeDat = readData(exptParams['dataFile'],numCohorts,exptParams['exCare'],
                                  exptParams['careFile'],verbose=verboseMod) # ,maxWeeks=15)

    simTime = exptParams['timeLast'] - exptParams['timeZero'] - 1  ## note -1 for fenceposts...

    ## data array (cumulative weekly deaths as fraction of population)
    obsDeath = np.copy( deathCumulativeDat[ :,exptParams['timeZero']:exptParams['timeLast'] ].transpose() )
    obsDeath /= N

    ## filter matrix
    fltrDeath = fltrClasses(['Im'],numCohorts,model_spec,verbose=False)

    ### priors for epidemiological and NPI ("control") params

    priorsAll = getPriorsEpiThreeStage(modParams,exptParams['inferBetaNotAi'])

    stepControl = True      ## step-like-NPI
    controlWithLog = False  ## keep as False (not used)
   
    if stepControl : 
        print('** using getPriorsControl')
        [priorsCon,interventionFn]=getPriorsControl(numCohorts,daysPerWeek,
                                                #inferBetaNotAi=exptParams['inferBetaNotAi'],
                                                widthPriorMean=12.0, ## days
                                               )
#    elif controlWithLog :
#        print('** using getPriorsControlGMobLog')
#        [priorsCon,interventionFn]=getPriorsControlGMobLog(numCohorts,daysPerWeek,
#                                                #inferBetaNotAi=exptParams['inferBetaNotAi'],
#                                                widthPriorMean=12.0, ## days
#                                               )
    else :
        print('** using getPriorsControlGMob')
        [priorsCon,interventionFn]=getPriorsControlGMob(numCohorts,daysPerWeek,
                                                #inferBetaNotAi=exptParams['inferBetaNotAi'],
                                                widthPriorMean=12.0, ## days
                                               )
    ## add these NPI priors to priorsAll
    priorsAll.update(priorsCon)

    if verboseMod : pprint.pprint(priorsAll)

    # dictionary with initial guess for control params (prior mean)
    guessCon = dict( [ [key,np.array(priorsCon[key]['mean']) ]
                            for key in priorsCon ] )
    if verboseMod : pprint.pprint(guessCon)

    # ### initial condition prior and fltr (name is ..LinMode but it is the full prior)

    initPriorsLinMode = getInitPriors(model_spec,numCohorts,N,exptParams['freeInitPriors'],fixR=True)
    if verboseMod : pprint.pprint(initPriorsLinMode)

    # ### pyRoss estimator object setup

    stepParam=7  ## technical param for ODE solving
    estimator = pyross.inference.Spp(model_spec, modParams,
                                     numCohorts, fi, int(N), stepParam,
                                     parameter_mapping=paramMap,
                                     rtol_det=exptParams['estimatorTol'])
    estimator.set_lyapunov_method('euler')

    ## prior (time-dependent) contact matrix, including NPI
    guessCM = contactBasis.intervention_custom_temporal( interventionFn, **guessCon)
    estimator.set_contact_matrix( guessCM )

    if verboseMod :
        tVals = np.linspace(0,10,100)
        plt.plot(tVals,[guessCM(tt)[3,4] for tt in tVals])
        plt.plot(tVals,[guessCM(tt)[6,7] for tt in tVals])
        plt.show() ; plt.close()

    return [ numCohorts, fi, N, Ni, model_spec, estimator, contactBasis, interventionFn,
             modParams, priorsAll, initPriorsLinMode, obsDeath, fltrDeath, simTime, deathCumulativeDat ]
