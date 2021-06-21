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


postFigFileName = 'figPostHistos_pop1e4.pdf'
trajFigFileName = 'figTraj_pop1e4.pdf'
mapFigFileName = 'figInfTraj_pop1e4.pdf'
scatFigFileName = 'figScatter_pop1e4.pdf'


useTex = True

if useTex :
  plt.rc('text', usetex=True)
  plt.rcParams.update({'font.size': 26})
  plt.rcParams['font.family'] = 'serif'

import synth_fns

## for dataFiles : needs a fresh value in every notebook
fileRoot = 'dataSynthInfTest-pop1e4'

## total population
popN = 1e4

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

ipFile = fileRoot+'-stochTraj.npy'
syntheticData = np.load(ipFile)
print('loading trajectory from',ipFile)
Nf_start = synth_fns.get_start_time(syntheticData, popN, fracDeaths)
print('inf starts at timePoint',Nf_start)


fineDataPlot = 1  ## this has to do with time rescaling

## note this fig is not used in the paper(!)
def plotTraj(trajFigFileName,M,data_array,Nf_start,Tf_inference,fineData):
    fig,axs = plt.subplots(1,2, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.08,bottom=0.2,wspace=0.4,right=0.97)
    ax = axs[0]

    t = np.linspace(0, Tf/fineDataPlot, Nf)

    # plt.plot(t, np.sum(data_array[:, :M], axis=1), '-o', label='S', lw=4)
    ax.plot(t, np.sum(data_array[:, M:2*M], axis=1), '-', label='$E$', lw=4)
    ax.plot(t, np.sum(data_array[:, 2*M:3*M], axis=1), '-', label='$I$', lw=4)
    ax.plot(t, np.sum(data_array[:, 3*M:4*M], axis=1), '-', label='$D$', lw=4)
    #plt.plot(t, N-np.sum(data_array[:, 0:4*M], axis=1), '-o', label='Rec', lw=2)

    ax.axvspan(Nf_start/fineDataPlot, (Nf_start+Tf_inference)/fineDataPlot,alpha=0.25, color='dodgerblue')
    ax.legend(handlelength=0.2,frameon=False)
    ax.set_xlabel('time')

    #plt.savefig(trajFig1)
    #plt.show()

    ax = axs[1]
    ax.plot(t[1:],np.diff(np.sum(data_array[:, 3*M:4*M], axis=1)),'o',label='death increments', lw=1)
    ax.axvspan(Nf_start/fineDataPlot, (Nf_start+Tf_inference)/fineDataPlot,alpha=0.25, color='dodgerblue')
    #ax.legend(loc='upper right') ; # plt.show()
    ax.set_ylabel('daily deaths')
    ax.set_xlabel('time')
    
    #yTicks = [0,5e4]
    #yTickLabels = ['0','$5\\times 10^4$']
    #ax.set_yticks(yTicks)
    #ax.set_yticklabels(yTickLabels)

#    ax = axs[1]
#    ax.plot(t,np.sum(data_array[:, 3*M:4*M], axis=1),'o-',label='deaths',ms=3)
#    ax.legend() ;

    plt.savefig(trajFigFileName)

plotTraj(trajFigFileName,cohortsM,syntheticData,Nf_start,Tf_inference,fineData)

[estimator,fltrDeath,obsData,trueInit] = synth_fns.get_estimator(isHighAccuracy,model_dict,syntheticData, popN, Nf_start, Nf_inference,)

## compute log-likelihood of true params
logpTrue = -estimator.minus_logp_red(parameters_true, trueInit, obsData, fltrDeath, Tf_inference,
                                     contactMatrix, tangent=False)
print('**logLikTrue',logpTrue,'\n')

ipFile = fileRoot+'-mcmc.pik'
print('ipf',ipFile)
with open(ipFile, 'rb') as f:
    [infResult,result_mcmc] = pickle.load(f)


def plotMAP(mapFileName,res,data_array,M,N,estimator,Nf_start,Tf_inference,fineData):
    #print('**beta(bare units)',res['params_dict']['beta']*fineData)
    #print('**logLik',res['log_likelihood'],'true was',logpTrue)
    #print('\n')
    #print(res)

    fig,axs = plt.subplots(3,1, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k',sharex=True)
    plt.subplots_adjust(left=0.15,bottom=0.09,top=0.95,wspace=0.3,right=0.73,hspace=0.25)
    #plt.subplots_adjust(left=0.13,bottom=0.06,top=0.99,wspace=0.3,right=0.76)
    t = np.linspace(0, Tf/fineDataPlot, Nf)

    ax = axs[0]
    axs[-1].set_xlabel('time / days')

    datArgs = {'ms':6,'alpha':0.6,'linestyle':'none','marker':'o'}
    mapArgs = {'lw':4}
    
    cols = ['C0','C1','C2','C3','C4']  ## E I D S R

    ## DATA
    ax.plot(t, np.sum(data_array[:, M:2*M], axis=1), **datArgs, color=cols[0], label='$E$') # 'o', lw=2, ms=4,alpha=0.7)
    ax.plot(t, np.sum(data_array[:, 2*M:3*M], axis=1), **datArgs, color=cols[1], label='$I$') # , lw=2, ms=4,alpha=0.7)
    ax.plot(t, np.sum(data_array[:, 3*M:4*M], axis=1), **datArgs, color=cols[2], label='total $D$')#  , lw=2, ms=4,alpha=0.7)


    ## MAP
    tt = np.linspace(Nf_start, Tf, Nf-Nf_start,)/fineDataPlot
    xm = estimator.integrate(res['x0'], Nf_start, Tf, Nf-Nf_start, dense_output=False)
    
    ax.plot(tt, np.sum(xm[:, M:2*M], axis=1),   '-',color=cols[0], **mapArgs)
    ax.plot(tt, np.sum(xm[:, 2*M:3*M], axis=1), '-', color=cols[1], **mapArgs)
    ax.plot(tt, np.sum(xm[:, 3*M:4*M], axis=1), '-', color=cols[2], **mapArgs)

    ax.axvspan(Nf_start/fineData, (Nf_start+Tf_inference)/fineDataPlot,alpha=0.3, color='dodgerblue')
    ax.legend(handlelength=0.2,bbox_to_anchor=(1, 1.0),frameon=False,loc='upper left')

    ax = axs[1]
    ax.plot(t[1:], np.diff(np.sum(data_array[:, 3*M:4*M], axis=1)), color=cols[2], label='daily $D$',**datArgs,)
    ax.plot(tt[1:], np.diff(np.sum(xm[:, 3*M:4*M], axis=1)),color=cols[2], **mapArgs)
    ax.axvspan(Nf_start/fineData, (Nf_start+Tf_inference)/fineDataPlot,alpha=0.3, color='dodgerblue')
    ax.legend(handlelength=0.2,bbox_to_anchor=(1, 1.0),frameon=False,loc='upper left')
    ax.set_yticks([0,5,10])

    ax = axs[2]

    ax.plot(t, np.sum(data_array[:, :M], axis=1), label='$S$', color=cols[3], **datArgs)
    ax.plot(t, N-np.sum(data_array[:, 0:4*M], axis=1), label='$R$',  color=cols[4], **datArgs)

    #infResult = res
    #tt = np.linspace(Nf_start, Tf, Nf-Nf_start,)/fineDataPlot
    #xm = estimator.integrate(res['x0'], Nf_start, Tf, Nf-Nf_start, dense_output=False)
    ax.plot(tt, np.sum(xm[:, :M], axis=1),  color=cols[3], **mapArgs)
    ax.plot(tt, N-np.sum(xm[:, :4*M], axis=1), color=cols[4], **mapArgs)

    ax.axvspan(Nf_start/fineDataPlot, (Nf_start+Tf_inference)/fineData,alpha=0.3, color='dodgerblue')
    ax.legend(handlelength=0.2,bbox_to_anchor=(1, 1.0),frameon=False,loc='upper left')
    
    plt.savefig(mapFileName)

plotMAP(mapFigFileName,infResult,syntheticData,cohortsM,popN,estimator,Nf_start,Tf_inference,fineDataPlot)


def plotPosteriorHistos(postFileName,estimator,obsData, fltrDeath, Tf_inference,param_priors,
                   init_priors,contactMatrix,
                   infResult,result_mcmc,parameters_true,trueInit,empColor='lightblue') :
                  
    ## used for prior pdfs
    (likFun,priFun,dimFlat) = pyross.evidence.latent_get_parameters(estimator,
                                        obsData, fltrDeath, Tf_inference,
                                        param_priors, init_priors,
                                        contactMatrix,
                                        #intervention_fun=interventionFn,
                                        tangent=False,
                                      )
                                      
    fig,axs = plt.subplots(2,2,figsize=(8,7))
    plt.subplots_adjust(left=0.08,bottom=0.13,hspace=0.35,right=0.97,top=0.99)
    
    mapArgs = {'ms':11,'color':'blue','marker':'*','label':'MAP','linestyle':'none'}
    trueArgs = {'ms':8,'color':'red','marker':'o','label':'true','linestyle':'none'}
    priArgs = {'lw':2.5,'color':'darkgreen','label':'prior'}
    
    histAlpha = 0.8
    
    ax = axs[0,0]
                     
    betaRangeParam = 0.8
                     
    xVals = np.linspace(parameters_true['beta']*(1-betaRangeParam),
                        parameters_true['beta']*(1+betaRangeParam),100)

    betas = [ rr['params_dict']['beta'] for rr in result_mcmc ]
    ax.hist(betas,density=True,color=empColor,alpha=histAlpha) # ,label='posterior')
    # decide where to put the dots for MAP/true
    stdBeta = np.std(betas)
    yVal=0.03/stdBeta
    ax.plot([infResult['params_dict']['beta']],[2*yVal], **mapArgs) # ,'bs',label='MAP')
    ax.plot([parameters_true['beta']],[yVal],**trueArgs) # 'ro',label='true')


    ## this is a bit complicated, it just finds the prior for beta from the infResult
    var='beta'
    jj = infResult['param_keys'].index(var)
    xInd = infResult['param_guess_range'][jj]
    #print(jj,xInd)
    pVals = []
    for xx in xVals :
        flatP = np.zeros( dimFlat )
        flatP[xInd] = xx
        pdfAll = np.exp( priFun.logpdf(flatP) )
        pVals.append( pdfAll[xInd] )
    ax.plot(xVals,pVals,**priArgs) # color='darkgreen',label='prior')


    ax.set_xlabel('$\\beta$')
    ax.yaxis.set_ticklabels([])

    labs=['$S_0/\Omega$','$E_0/\Omega$','$I_0/\Omega$']
    nPanel=3
    for ii in range(nPanel) :
        ax = axs[int((ii+1)/2),(ii+1)%2]

        xs = np.array( [ rr['x0'][ii] for rr in result_mcmc ] )/popN
        # decide where to put the dots for MAP/true
        stdX = np.std(xs)
        yVal = 0.03/stdX
        ax.hist(xs,color=empColor,alpha=histAlpha,density=True)
        ax.plot([infResult['x0'][ii]/popN],2*yVal, **mapArgs)
        ax.plot([trueInit[ii]/popN],yVal,**trueArgs) # 'ro',label='true')

        ## this is a bit complicated, it just finds the prior for beta from the infResult
        ## axis ranges
        xMin = np.min(xs)*0.8
        xMax = np.max(xs)*1.2
        
        if ii == 0 : xMin *= 0.8 ## make space for legend

        xVals = np.linspace(xMin,xMax,100)


        ## this ID is a negative number because the init params are the end of the 'flat' param array
        paramID = ii-nPanel
        pVals = []
        for xx in xVals :
            flatP = np.zeros( dimFlat )
            flatP[paramID] = xx
            pdfAll = np.exp( priFun.logpdf(flatP*popN) + np.log(popN) )
            pVals.append( pdfAll[paramID] )
        ax.plot(xVals,pVals,**priArgs) ## color='darkgreen',label='prior')

        ax.set_xlabel(labs[ii])
        #ax.set_ylabel('pdf')
        ax.yaxis.set_ticklabels([])
        
        if ii>0 : ax.set_xlim(left=0,right=xMax)
        else : ax.set_xlim(left=xMin,right=xMax)

    for ax in axs[:,0] : ax.set_ylabel('pdf')
    axs[0,1].legend(handlelength=0.2,frameon=False,loc='upper left',bbox_to_anchor=(-0.03, 1.0))
    plt.savefig(postFileName)


[param_priors,init_priors] = synth_fns.get_priors(model_dict,betaPriorOffset,betaPriorLogNorm,fracDeaths,estimator)

plotPosteriorHistos(postFigFileName,estimator,obsData, fltrDeath, Tf_inference,
                   param_priors, init_priors,contactMatrix,
                   infResult,result_mcmc,parameters_true,trueInit,empColor='dodgerblue')

def plotScatter(figFileName,result_mcmc,popN) :
    fig,axs = plt.subplots(1,2,figsize=(8,4.5))
    plt.subplots_adjust(left=0.14,bottom=0.2,wspace=0.44,right=0.98,top=0.95)


    ## where are the SEI compartment vals in the rr['x0'] array ?
    [ indS,indE,indI ] = [0,1,2]

    sVals = np.array( [ rr['x0'][indS]/popN for rr in result_mcmc ] )
    eVals = np.array( [ rr['x0'][indE]/popN for rr in result_mcmc ] )
    iVals = np.array( [ rr['x0'][indI]/popN for rr in result_mcmc ] )

    betaVals = [ rr['params_dict']['beta'] for rr in result_mcmc ]

    ax = axs[1]

    ax.plot(eVals,iVals,'o',ms=3)
    ax.set_xlabel('$E_0/\\Omega$')
    ax.set_ylabel('$I_0/\\Omega$')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    pearson = np.corrcoef(eVals,iVals)[0,1]

    #spac = 0.05 ## spacing
    #ax.set_yticks([spac*i for i in range(0,1+int(np.max(iVals)/spac))])
    #ax.set_xticks([spac*i for i in range(0,1+int(np.max(eVals)/spac))])

    ax.text(0.009,0.004,'$R^2 ='+'{:.3f}'.format(pearson**2)+'$')

    print('R^2 of E0 vs I0\n', pearson**2)

    ax = axs[0]

    xx = betaVals
    yy = 1-(sVals+eVals+iVals)
    ax.plot(xx,yy,'o',ms=3)
    pearson = np.corrcoef(xx,yy)[0,1]

    ax.set_xlabel('$\\beta$')
    ax.set_ylabel('$R_0/\\Omega$')

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    
    ax.text(0.003,0.24,'$R^2 ='+'{:.3f}'.format(pearson**2)+'$')

    print('R^2 of beta vs R\n', pearson**2)
    #print('pearson \n',np.corrcoef(betaVals,1-(sVals+eVals+iVals)) )

    plt.savefig(figFileName)


plotScatter(scatFigFileName,result_mcmc,popN)
