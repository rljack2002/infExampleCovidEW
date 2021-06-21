#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import pickle
import pprint
import time

import pyross
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from matplotlib import rc;


#postFigFileName = 'figPostHistos_pop1e8.pdf'
#trajFigFileName = 'figTraj_pop1e8.pdf'
#mapFigFileName = 'figInfTraj_pop1e8.pdf'


useTex = True

if useTex :
  plt.rc('text', usetex=True)
  plt.rcParams.update({'font.size': 20})
  plt.rcParams['font.family'] = 'serif'

import synth_fns


## total population
popN = 1e6

## tau-leaping param, take this negative to force gillespie
## or set a small value for high-accuracy tau-leap (eg 1e-4 or 1e-5)
leapEps = 1e-5

## do we use small tolerances for the likelihood computations? (use False for debug etc)
isHighAccuracy = True

# absolute tolerance for logp for MAP
inf_atol = 1.0

## prior mean of beta, divided by true value (set to 1.0 for the simplest case)
betaPriorOffset = 1.0
betaPriorLogNorm = False

## setup model etc ( copied from synthInfTest-pop1e8.ipynb )

model_dict = synth_fns.get_model(popN)

model_spec = model_dict['mod']
contactMatrix = model_dict['CM']
parameters_true = model_dict['params']
cohortsM = model_dict['cohortsM']
Ni = model_dict['cohortsPop']

## total trajectory time (bare units)
Tf_bare = 20
## total inf time
Tf_inf_bare = 5

## inference period starts when the total deaths reach this amount (as a fraction)
fracDeaths = 2e-3 # int(N*200/1e5)

## hack to get higher-frequency data
## how many data points per "timestep" (in original units)
fineData = 4

## this assumes that all parameters are rates !!
for key in parameters_true:
    #print(key,parameters_true[key])
    parameters_true[key] /= fineData

#Tf = Tf_bare * fineData;
#Nf = Tf+1
#
#Tf_inference = Tf_inf_bare * fineData
#Nf_inference = Tf_inference+1

def getResults(fileRoot,minSeed) :

#    ipFile = fileRoot+'-run'+str(0)+'-stochTraj'+str(minSeed)+'.npy'
#    syntheticData = np.load(ipFile)
#    print('loading trajectory from',ipFile)
#    Nf_start = synth_fns.get_start_time(syntheticData, popN, fracDeaths)
#    print('inf starts at timePoint',Nf_start)

    runVals = [0,1]

    allResultsInf = []
    allResultsMC = []
    for runVal in runVals :
        ipFile = fileRoot+'-run'+str(runVal)+ "-mcmcAll.pik"
        print('ipf',ipFile)
        with open(ipFile, 'rb') as f:
            [loadInf,loadMC]= pickle.load(f)

        print('** read',len(loadInf),'data sets')

        allResultsInf += loadInf
        allResultsMC += loadMC

    print('** tot',len(allResultsInf),'data sets ( check',len(allResultsMC),')')


    return [allResultsInf,allResultsMC]


def computeBetaStats(allResultsMC,allResultsInf,printMe=True) :

    betaStats = []
    for trajIndex,result_mcmc in enumerate(allResultsMC) :
        betas = [ rr['params_dict']['beta'] for rr in result_mcmc ]
        postMeanBeta = np.mean(betas)
        postStdBeta = np.std(betas)
        postCIBeta = [ np.percentile(betas,2.5) , np.percentile(betas,97.5)]

        betaStats += [{'m':postMeanBeta,'s':postStdBeta,'c':postCIBeta,
                       'map':allResultsInf[trajIndex]['params_dict']['beta']}]
        
        if printMe :
          print("post: mean {m:.4f} std {s:.4f} CI95 {l:.4f} {u:.4f}".format(m=postMeanBeta,
                                                                         s=postStdBeta,
                                                                         l=postCIBeta[0],u=postCIBeta[1]))

    meanPostMean = np.mean(np.array([ b['m'] for b in betaStats ]))
    stdPostMean = np.std(np.array([ b['m'] for b in betaStats ]))
    errPostMean = stdPostMean/np.sqrt(len(allResultsInf)-1)
    meanPostStd = np.mean(np.array([ b['s'] for b in betaStats ]))
    meanPostCI = [ np.mean(np.array([ b['c'][ii] for b in betaStats ])) for ii in [0,1] ]
    meanMAP = np.mean(np.array([ b['map'] for b in betaStats ]))

    if printMe :
        print('\n')
        print('** true {:.4f}'.format(parameters_true['beta']))
        print('** meanPostMAP {:.4f}'.format(meanMAP))
        print('** meanPostMean {:.4f}'.format(meanPostMean))
        print('** stdPostMean {:.4f} stderr {:.4f} (n {:d})'.format(stdPostMean,
                                                           stdPostMean/np.sqrt(len(allResultsInf)-1) ,
                                                           len(allResultsInf)
                                                          ))
        print('** meanPostCI {:.4f} {:.4f}'.format(meanPostCI[0],meanPostCI[1]))
        print('** meanPostStd  {:.4f}'.format(meanPostStd))

    return [betas,meanPostMean,stdPostMean,errPostMean,meanPostCI]


minSeed = 19
rootList = ['dataSynthInfQuality-pop1e4','dataSynthInfQuality-pop1e5','dataSynthInfQuality-pop1e6']
popList = [1e4,1e5,1e6]

yVals = []
barVals = []
ciVals = []

for jj,fileRoot in enumerate(rootList) :
    print('***',fileRoot)
    [fileResultsInf,fileResultsMC] = getResults(fileRoot,minSeed)
    [betas,meanPostMean,stdPostMean,errPostMean,meanPostCI] = computeBetaStats(fileResultsMC,fileResultsInf)
    
    yVals += [meanPostMean]
    barVals += [errPostMean]
    ciVals += [meanPostCI]
    
fig,ax = plt.subplots(1,1,figsize=(7, 4))
plt.subplots_adjust(left=0.15,right=0.95,bottom=0.2,top=0.95)

ax.set_xscale('log')
ax.set_xlabel('population $N$')
ax.set_ylabel('$\\beta$')


ax.errorbar(popList,yVals,yerr=barVals,fmt='o',label='average posterior mean')

ax.fill_between(popList,[c[0] for c in ciVals],[c[1] for c in ciVals],
                color='dodgerblue',alpha=0.2,label='average posterior CI')

ax.plot([np.min(popList),np.max(popList)],[parameters_true['beta'],parameters_true['beta']],
         linestyle='dashed',color='red')


ax.set_ylim(bottom=0.0,top=2.0*parameters_true['beta'])


ax.legend(handlelength=0.2,frameon=False)
plt.savefig('figQuality.pdf')
