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


foreFigFileName = 'figForecast_pop1e6.pdf'
#trajFigFileName = 'figTraj_pop1e4.pdf'
#mapFigFileName = 'figInfTraj_pop1e4.pdf'


useTex = True

if useTex :
  plt.rc('text', usetex=True)
  plt.rcParams.update({'font.size': 22})
  plt.rcParams['font.family'] = 'serif'

import synth_fns

## for dataFiles : needs a fresh value in every notebook
fileRoot = 'dataSynthInfTest-pop1e6-win'

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
betaPriorOffset = 0.8
betaPriorLogNorm = False

## mcmc
mcSamples = 8000
nProcMCMC = 2 # None ## take None to use default but large numbers are not efficient in this example

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

Tf = Tf_bare * fineData;
Nf = Tf+1

Tf_inference = Tf_inf_bare * fineData
Nf_inference = Tf_inference+1

ipFile = fileRoot+'5-stochTraj.npy'
syntheticData = np.load(ipFile)
print('loading synthetic trajectory from',ipFile)
Nf_start = synth_fns.get_start_time(syntheticData, popN, fracDeaths)
print('inf starts at timePoint',Nf_start)

fineDataPlot = 1


[estimator,fltrDeath,obsData,trueInit] = synth_fns.get_estimator(isHighAccuracy,model_dict,syntheticData, popN, Nf_start, Nf_inference,)

## compute log-likelihood of true params
logpTrue = -estimator.minus_logp_red(parameters_true, trueInit, obsData, fltrDeath, Tf_inference,
                                     contactMatrix, tangent=False)
print('**logLikTrue',logpTrue,'\n')

[param_priors,init_priors] = synth_fns.get_priors(model_dict,betaPriorOffset,betaPriorLogNorm,fracDeaths,estimator)


winRange = [ xx for xx in range(1,6) ]

allResultsInf = []
allResultsMC = []
allForeTraj = []

for win in winRange :
  ipFile = fileRoot+str(win)+'-mcmc.pik'
  print('ipf',ipFile)
  with open(ipFile, 'rb') as f:
      [infResult,result_mcmc] = pickle.load(f)
      allResultsInf += [infResult]
      allResultsMC += [result_mcmc]
      
  allForeTraj += [ np.load('dataSynthInfTest-pop1e6-win'+str(win)+'-foreTraj.npy') ]

time_points = np.linspace(0, Tf, Nf)

#toPlotTraj = [foreTraj_win1,foreTraj_win2,foreTraj_win3,foreTraj_win4,foreTraj_win5]

Tfi_vals = 4*np.array(winRange) # [4,8,12,16,20]
Nfi_vals = [ tt+1 for tt in Tfi_vals ]
nPanel = len(allForeTraj)

empColor='dodgerblue'

def plotMe(data_array,M):

    ## used for prior pdfs
    (likFun,priFun,dimFlat) = pyross.evidence.latent_get_parameters(estimator,
                                        obsData, fltrDeath, Tf_inference,
                                        param_priors, init_priors,
                                        contactMatrix,
                                        #intervention_fun=interventionFn,
                                        tangent=False,
                                      )

    fig,axs = plt.subplots(2,nPanel, figsize=(15, 7), sharey='row', sharex='row')
    plt.subplots_adjust(top=0.99,left=0.07,bottom=0.11,right=0.99,hspace=0.34)
    #plt.rcParams.update({'font.size': 14})
    #for x_start in samples:

    mapArgs = {'ms':4,'color':'blue','marker':'s'}

    axs[0,0].set_ylabel('daily deaths')
    axs[1,0].set_ylabel('pdf')

    labX,labY = [-0.4,0.9]
    axs[0,0].text(labX,labY,'(a)',transform=axs[0,0].transAxes)
    axs[1,0].text(labX,labY,'(b)',transform=axs[1,0].transAxes)

    for panelID in range(nPanel) :
    

        ## traj
        ax=axs[0,panelID]

        for traj in allForeTraj[panelID]:
            incDeaths = np.diff( np.sum(traj[:, 3*M:4*M], axis=1) )
            ax.plot(time_points[1+Tfi_vals[panelID]+Nf_start:], incDeaths, color='coral', alpha=0.2)

        incDeathsObs = np.diff( np.sum(data_array[:, 3*M:4*M], axis=1) )

        ax.plot(time_points[1:],incDeathsObs, 'ko', label='True D',ms=4)
        ax.axvspan(Nf_start, Tfi_vals[panelID]+Nf_start,
                   label='Used for inference',
                   alpha=0.3, color='dodgerblue')
        ax.set_xlim([0, Tf])
        
        ax.set_xlabel('time / days')
        
        ## beta histo
        ax=axs[1,panelID]
        
        mapArgs = {'ms':11,'color':'blue','marker':'*','label':'MAP','linestyle':'none'}
        trueArgs = {'ms':8,'color':'red','marker':'o','label':'true','linestyle':'none'}
        priArgs = {'lw':2.5,'color':'darkgreen','label':'prior'}
        
        betas = [ rr['params_dict']['beta'] for rr in allResultsMC[panelID] ]
        ax.hist(betas,density=True,color=empColor) # ,label='posterior')
        # decide where to put the dots for MAP/true
        stdBeta = np.std(betas)
        yVal=0.01/stdBeta
        ax.plot([allResultsInf[panelID]['params_dict']['beta']],[2*yVal], **mapArgs)
        ax.plot([parameters_true['beta']],[yVal],**trueArgs) # 'ro',label='true')

        ## prior
        betaRangeParam = 0.8
                     
        xVals = np.linspace(parameters_true['beta']*(1-betaRangeParam),
                        parameters_true['beta']*(1+betaRangeParam),100)

        
        ## this is a bit complicated, it just finds the prior for beta from the infResult
        var='beta'
        jj = allResultsInf[panelID]['param_keys'].index(var)
        xInd = allResultsInf[panelID]['param_guess_range'][jj]
        #print(jj,xInd)
        pVals = []
        for xx in xVals :
            flatP = np.zeros( dimFlat )
            flatP[xInd] = xx
            pdfAll = np.exp( priFun.logpdf(flatP) )
            pVals.append( pdfAll[xInd] )
        ax.plot(xVals,pVals,**priArgs) # color='darkgreen',label='prior')


        ax.set_xlabel('$\\beta$')


    axs[1,0].legend(handlelength=0.3,frameon=False)

    #plt.legend()
    plt.savefig(foreFigFileName)
    
plotMe(syntheticData,cohortsM)

