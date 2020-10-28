#!/usr/bin/env python
# coding: utf-8

## python script for making Figs
## NPI-with0-easing
## note we run deterministic dynamics to get MAP trajectory (etc), requires stable pyross...

## output  in ../finalFigs

### easeForecast.pdf : det and stoch forecast
### foreWin.pdf : det forecasts with increasing time window

## various posterior dists (dep on inf window):
#### winEaseBKDE.pdf
#### winEaseBKDE-offset.pdf
#### winLateBKDE-offset.pdf

## various posterior mean (dep on inf window):
### winGamma.pdf
### winLock.pdf
### winBeta_aF.pdf

## option to skip some figs, saves a few seconds for efficient editing
doForecast = True
doKDEs = True

import numpy as np
from   matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm
import time

import pyross
import pickle
import pprint

import scipy.stats

from uk_v2a_fns import *
import expt_params_local
import model_local

## filename root for stored results
pikFileRoot = "ewMod"

## where to put figs
figPath = "../finalFigs/"

## time unit is one week
daysPerWeek = 7.0

## GET model params
## these are params that might be varied in different expts
exptParams = expt_params_local.getLocalParams()

print('** exptParams')
pprint.pprint(exptParams)

## LOAD model
loadModel = model_local.loadModel(exptParams,daysPerWeek,verboseMod=False,yesComet=False)

[ numCohorts, fi, N, Ni, model_spec, estimator, contactBasis, interventionFn,
   modParams, priorsAll, initPriorsLinMode, obsDeath, fltrDeath,
   simTime, deathCumulativeDat ] = loadModel

## we quote figsize numbers as if they were cm (in fact they are inches)
## this means 20 is ok as default font
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'serif'
plt.rc('text', usetex=True)

ipFile = pikFileRoot + "-sample-post.pik"
print('ipf',ipFile)
with open(ipFile, 'rb') as f:
    [trajStochPost] = pickle.load(f)

ipFile = pikFileRoot + "-inf.pik"
print('ipf',ipFile)
with open(ipFile, 'rb') as f:
    [infResult,tt] = pickle.load(f)

ipFile = pikFileRoot + "-result_mcmc.pik"
print('ipf',ipFile)
with open(ipFile, 'rb') as f:
    [result_mcmc] = pickle.load(f)
    
## cohort age ranges
cohRanges = [ [x,x+4] for x in range(0,75,5) ]
cohLabs = ["{l:d}-{u:d}".format(l=low,u=up) for [low,up] in cohRanges ]
cohLabs.append("75+")

## used for prior pdfs
(likFun,priFun,dimFlat) = pyross.evidence.latent_get_parameters(estimator,
                                    obsDeath, fltrDeath, simTime,
                                    priorsAll,
                                    initPriorsLinMode,
                                    generator=contactBasis,
                                    intervention_fun=interventionFn,
                                    tangent=False,
                                  )


## FIG : weekly deaths and forecast, det vs stoch forecast
## for weekly deaths
def diffs(dd) :
    op=[]
    for ii,di in enumerate(dd[1:]) : op.append( di-dd[ii] )
    return op


runTime = 10 
simTime = 7

nSel = 20  ## how many traj for each CM


for pp in ['left','right','top','bottom','wspace']:
  print('subP default',pp,plt.rcParams['figure.subplot.'+pp])

fig,axs = plt.subplots(1,2,figsize=(8.5,4.0),sharey=True)
plt.subplots_adjust(wspace=0.24)

indClass = model_spec['classes'].index('Im')
tValsMAP = np.linspace(1,simTime,simTime)
#totClass = N*np.sum(trajMAP[:,indClass*numCohorts:(indClass+1)*numCohorts],axis=1)
totDeathObs = np.sum(deathCumulativeDat[:,:(runTime+1)],axis=0)

ax = axs[0]
tVals = np.linspace(1,runTime,runTime)

## MAP traj
estimator.set_params(infResult['params_dict'])
estimator.set_contact_matrix( contactBasis.intervention_custom_temporal( interventionFn,
                                                                         **infResult['control_params_dict'])
                            )
trajMAPforecast = estimator.integrate( infResult['x0'], 0, runTime, runTime+1)


## trajectory samples (deterministic)
nPlot = 40
allTraj = []
for ii,rr in enumerate(result_mcmc[-nPlot:]) :
    estimator.set_params(rr['params_dict'])
    estimator.set_contact_matrix( contactBasis.intervention_custom_temporal( interventionFn,
                                                                             **rr['control_params_dict'])
                                )
    mytraj = estimator.integrate( rr['x0'], 0, runTime, runTime+1)
    allTraj.append(mytraj)

cc=model_spec['classes'].index('Im')
for ii,mytraj in enumerate(allTraj) :
    dMod = N * np.sum(mytraj[:,(cc*numCohorts):((cc+1)*numCohorts)],axis=1)
    lab=None
    ax.plot(tVals,diffs(dMod), '-', label=lab ,
                               color='C0' , alpha=0.4, lw=1 )
## for legend (with alpha=1)
ax.plot([],[],'-', label='model' , color='C0',alpha=1.0 ,lw=1)

dMod = N * np.sum(trajMAPforecast[:,(cc*numCohorts):((cc+1)*numCohorts)],axis=1)
ax.plot(tVals,diffs(dMod), '--', label='MAP' , color='black' , alpha=1.0, linewidth=2 )

dObs = np.sum(deathCumulativeDat[0:numCohorts,:runTime+1].transpose(), axis=1)
ax.plot(tVals,diffs(dObs), 'o', label='data' ,color='C1' )

ax.set_xticks( [xx for xx in range(0,11,2) ])

ax.set_ylabel('weekly deaths')
ax.set_xlabel('time (weeks)')
ax.tick_params(labelleft=True)
ax.axvspan(1, simTime,alpha=0.2, color='silver',
           label='inf window')

abLab = 0.05
abLabV = 0.96

ax.text(abLab, abLabV, '(a)', transform=ax.transAxes,
    va='top')

# note legend is done by second panel

ax = axs[1]
tVals = np.linspace(1,runTime,runTime)

tValsFore = np.linspace(1+simTime,runTime,exptParams['forecastTime'])
tValsAll = np.append(tValsMAP,tValsFore)

for ii,dataStoch in enumerate(trajStochPost) :
        totClassStoch = np.sum(dataStoch[:,indClass*numCohorts:(indClass+1)*numCohorts],axis=1)
        trajAll = np.append(totDeathObs[:simTime],totClassStoch)

        #if ii == 0 : lab='model ({n:d} samples)'.format(n=len(trajStochPost))
        #else : lab=None
        lab=None
        
        ax.plot( tValsAll,diffs(trajAll),'-',lw=1,label=lab,ms=4,color='C0',alpha=0.4)
        if False : print(diffs(trajAll)[-3])

## for legend
ax.plot([],[],'-',  label='model' , color='C0', alpha=1.0 ,lw=1)
ax.plot([],[],'--', label='MAP' ,   color='black' , alpha=1.0, linewidth=2 )

plt.xlabel('time (weeks)')
#plt.ylabel('weekly deaths')

ax.axvspan(1, simTime,alpha=0.2, color='silver',
           label='inf window')

ax.set_xticks( [xx for xx in range(0,11,2) ])

dObs = np.sum(deathCumulativeDat[0:numCohorts,:runTime+1].transpose(), axis=1)
ax.plot(tVals,diffs(dObs), 'o', label='data' ,color='orange', ms=5 )

## slightly weird legend across both panels
ax.legend(handlelength=1.0,loc='upper left',
           bbox_to_anchor=(-0.45, 1.02),framealpha=1.0)

abLab = 0.85
ax.text(abLab, abLabV, '(b)', transform=ax.transAxes, va='top')

plt.savefig(figPath+'easeForecast.pdf',bbox_inches='tight')
#plt.show(fig)
plt.close()


### now we vary the inference window, values 7,8,9,10

datArgs = {  'lw' : 2  , 'ms' : 3}
priArgs = { 'label' : 'prior', 'lw' : 3 , 'color' : 'gray' }

## colors for the different values
datCols = [ 'forestgreen','steelblue','dodgerblue','blue']

### LOAD data for different inference windows

simTimeVals = [7,8,9,10]
suffs = [""]+["-tWin{x:02d}".format(x=x) for x in [9,10,11] ]
print(suffs)

def loadResMCMC(suff) :
    ipFile = pikFileRoot + suff + "-result_mcmc.pik"
    print('ipf',ipFile)
    with open(ipFile, 'rb') as f:
        [res] = pickle.load(f)
    return res

allRes = [ loadResMCMC(ss) for ss in suffs ]

print('loaded samples',[len(rr) for rr in allRes])

if not doForecast :
    print('skipping forecast')   ## takes a few seconds
else:
    ii = 0

    runTime = 10  ## always fixed

    collateTraj = []

    ## get param samples and run dynamics
    for winID in range(len(allRes)) :
        print('run forecast',winID)
        nPlot = 40
        allTraj = []
        for ii,rr in enumerate(allRes[winID][-nPlot:]) :
            estimator.set_params(rr['params_dict'])
            estimator.set_contact_matrix( contactBasis.intervention_custom_temporal( interventionFn,
                                                                                     **rr['control_params_dict'])
                                        )
            mytraj = estimator.integrate( rr['x0'], 0, runTime, runTime+1)
            allTraj.append(mytraj)
            
        collateTraj.append(allTraj)
        
    ## FIGURE

    fig,axs = plt.subplots(1,len(allRes),figsize=(16,4),sharey=True)
    plt.subplots_adjust(left=0.07,right=0.85,bottom=0.19)

    for winID in range(len(allRes)) :
        ax=axs[winID]
        allTraj = collateTraj[winID]

        tVals = np.linspace(1,runTime,runTime)

        cc=model_spec['classes'].index('Im')
        for ii,mytraj in enumerate(allTraj) :
            dMod = N * np.sum(mytraj[:,(cc*numCohorts):((cc+1)*numCohorts)],axis=1)
            #if ii == 0 : lab='model ({n:d} samples)'.format(n=nPlot)
            #else : lab=None
            lab=None
            ax.plot(tVals,diffs(dMod), '-', label=lab , color=datCols[winID] , alpha=0.4, lw=1 )
        ## for legend
        ax.plot([],[],'-', label='model', color=datCols[winID] , alpha=1.0, lw=1)

        dObs = np.sum(deathCumulativeDat[0:numCohorts,:runTime+1].transpose(), axis=1)
        ax.plot(tVals,diffs(dObs), 'o', label='data' ,color='C1' )

        ax.set_xticks(range(2,11,2))

        ax.set_xlabel('time (weeks)')
        #ax.tick_params(labelleft=True)
        ax.axvspan(1, simTimeVals[winID],alpha=0.2, color='silver',
                   label='inf window')

    axs[0].set_ylabel('weekly deaths')
    axs[-1].legend(bbox_to_anchor=(1, 1.0))

    plt.savefig(figPath+'foreWin.pdf')
    #plt.show(fig)
    plt.close()


## HELPER FUNC
##
## this is a very simple KDE with a boundary correction
## it is not very accurate but is better than just truncation + renormalisation of the whole density
## in any case we only use it for plotting so we mostly care that the result is not misleading
##   bdyMode should be 'renorm' or 'reflect'
def bdyKDE(data,bWidth,xMin=-np.Inf,xMax=np.Inf,bdyMode='renorm') :
    assert bdyMode in ['renorm','reflect'] , 'KDE : bad bdyMode'
    def gauss(x,m,s): return np.exp(-((x-m)/s)**2 / 2.0) / np.sqrt(2*np.pi*s*s)
    ## this will be the KDE for the prob density function at x
    def res(x) :
        if x > xMax or x < xMin :
            rr = 0.0
        else :
            ## data inside interval (should maybe throw error if any are outside)
            dataRed = [ xx for xx in data if xx<xMax and xx>xMin ]
            ## the std KDE is the sum of this list of small Gaussiasns
            rr = np.array( [ gauss(x,mm,bWidth) for mm in dataRed] )
            
            ## we put "image charges" outside the interval, which folds the probability back inside
            if bdyMode == 'reflect' :
                rr = np.sum(rr)
                rr += np.sum( [ gauss(x,2*xMax-mm,bWidth) for mm in dataRed if mm>xMax-3*bWidth] )
                rr += np.sum( [ gauss(x,2*xMin-mm,bWidth) for mm in dataRed if mm<xMin+3*bWidth] )
            
            ## each little gaussian is (re)normalised to 1 over the interval
            else : # bdyMode == 'renorm'
                rWidth = bWidth*np.sqrt(2.0)  ## rescale width for use with erf
                norms = [ scipy.special.erf((xx-xMin)/rWidth) + scipy.special.erf((xMax-xx)/rWidth) for xx in dataRed ]
                norms = 0.5 * np.array(norms)
                rr = np.sum(rr/norms)
            
            rr /= len(dataRed)  ## normalise
        ## function called 'res' returns the estimated pdf
        return rr
    ## function bdyKDE returns the function called res
    return res

if not doKDEs :
    print('skipping KDE plots')
else :
    ## we use some generic machinery to get the data for the relevant param
    ## (in this case 'r')
    ## last index is the name of the param, for the plot
    [dic,var,labText] = ['control_params_dict','easeFrac','$r$']

    ## bounds for KDE (etc)
    (xMin,xMax) = priorsAll[var]['bounds']
    xMaxIsBound = True
    xMinIsBound = True

    ## for KDE
    kernelWidth = 0.03  ## "by hand"
    xValsKDE = np.linspace(xMin,xMax,100)

    ## PRIOR
               
    ## magic to work out the index of this param in flat_params
    jj = allRes[0][0]['param_keys'].index(var)
    xInd = allRes[0][0]['param_guess_range'][jj]

    pVals = []
    xVals = np.linspace(xMin,xMax,100)
    for xx in xVals :
        flatP = np.zeros( dimFlat )
        flatP[xInd] = xx
        pdfAll = np.exp( priFun.logpdf(flatP) )
        pVals.append( pdfAll[xInd] )

    ## put a zero at the lower bound
    if xMinIsBound :
        pVals = [0.0,*pVals]
        xVals = np.append(xMin,xVals)
    if xMaxIsBound :
        pVals = [*pVals,0.0]
        xVals = np.append(xVals,xMax)

    ## PLOT (this version is NOT used)

    fig,axs = plt.subplots(1,1,figsize=(6,3.6))
    plt.subplots_adjust(right=0.61,bottom=0.19,top=0.96)
    ax = axs

    ## plot prior
    ax.plot(xVals,pVals,**priArgs)

    ## KDE

    for ii,res in enumerate(allRes) :
        ## note these are simple lists now (not lists of single-element vectors)
        dataSet = [ rr[ dic ][ var ] for rr in res ]
        #dataPoints = [ bb for bb in xValsKDE ]
        #kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=kernelWidth).fit( dataSet )
        #kPdf = np.exp( kde.score_samples( dataPoints ) )
        
        #renorm = np.trapz(kPdf,xValsKDE)
        #kPdf /= renorm
        
        bKDE = bdyKDE(dataSet,kernelWidth,xMax=xMax,xMin=xMin,bdyMode='renorm')
        kPdf = np.array( [bKDE(x) for x in xValsKDE ])
        
        ## show the upper bound...
        if xMaxIsBound :
            xValsPlot = np.append(xValsKDE,xMax)
            kPdf = np.append(kPdf,0)
        
        ax.plot(xValsPlot,kPdf,color=datCols[ii],label='$t_{\\rm inf}='+'{t:d}$'.format(t=simTimeVals[ii]),
                                            lw=2,)

    ax.legend(bbox_to_anchor=(1.03,1))
    ax.set_xlabel(labText)
    ax.set_ylabel('pdf')
    ax.set_xlim(0,0.55) ## hardcoded

    plt.savefig(figPath+"winEaseBKDE.pdf")
    #plt.show(fig) ;
    plt.close()

    ## ALTERNATIVE VERSION. (preferred) with vertical offset

    vOff = 8.0 ## vertical offset

    fig,axs = plt.subplots(1,1,figsize=(6,3.6))
    plt.subplots_adjust(right=0.61,bottom=0.19,top=0.96)
    ax = axs

    ## PRIOR
    ax.plot(xVals,pVals,**priArgs)

    kernelWidth = 0.03  ## "by hand"
    xValsKDE = np.linspace(xMin,xMax,100)

    for ii,res in enumerate(allRes) :
        ## note these are simple lists now (not lists of single-element vectors)
        dataSet = [ rr[ dic ][ var ] for rr in res ]
        
        bKDE = bdyKDE(dataSet,kernelWidth,xMax=xMax,xMin=xMin,bdyMode='renorm')
        kPdf = np.array( [bKDE(x) for x in xValsKDE ])
        
        ## show the upper bound...
        if xMaxIsBound :
            xValsPlot = np.append(xValsKDE,xMax)
            kPdf = np.append(kPdf,0)
        
    #    ax.plot(xValsPlot,kPdf+(1+ii)*vOff,color=datCols[ii],
    #                      label='$t_{\\rm inf}='+'{t:d}$'.format(t=simTimeVals[ii]),
    #                      lw=2,)

        ax.fill_between(xValsPlot,kPdf+(1+ii)*vOff,(1+ii)*vOff,color=datCols[ii],
                          label='$t_{\\rm inf}='+'{t:d}$'.format(t=simTimeVals[ii]),
                          alpha=0.7,
                          #lw=2,)
                          )
       
    ## magic to print the legend with reversed order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels),
              loc='upper left',bbox_to_anchor=(1.03,1))
              
    ax.set_xlabel(labText)
    ax.set_ylabel('pdf')
    ax.set_xlim(0,0.55) ## hardcoded
    ax.set_yticks([])

    plt.savefig(figPath+"winEaseBKDE-offset.pdf")
    #plt.show(fig) ;
    plt.close()


    ##  generic machinery again
    ## (in this case betaLateFactor, ie nu_L)
    ## last index is the name of the param, for the plot
    [dic,var,labText] = ['params_dict','betaLateFactor','$\\nu_{\\rm L}$']

    #print(priorsAll)

    ## bounds for KDE (etc)
    (xMin,xMax) = priorsAll[var]['bounds']
    xMaxIsBound = True
    xMinIsBound = True

    ## for KDE
    kernelWidth = 0.008  ## "by hand"
    xValsKDE = np.linspace(xMin,xMax,500)  ## lots of points because we don't show the whole range

    ## VERSION with offset

    vOff = 22.0 ## vertical offset

    fig,axs = plt.subplots(1,1,figsize=(3,3.6))
    plt.subplots_adjust(right=0.95,bottom=0.19,top=0.96)
    ax = axs

    ## PRIOR

    ## magic to work out the index of this param in flat_params
    jj = allRes[0][0]['param_keys'].index(var)
    xInd = allRes[0][0]['param_guess_range'][jj]

    pVals = []
    xVals = np.linspace(xMin,xMax,100)
    for xx in xVals :
        flatP = np.zeros( dimFlat )
        flatP[xInd] = xx
        pdfAll = np.exp( priFun.logpdf(flatP) )
        pVals.append( pdfAll[xInd] )

    ## put a zero at the lower bound
    if xMinIsBound :
        pVals = [0.0,*pVals]
        xVals = np.append(xMin,xVals)
    if xMaxIsBound :
        pVals = [*pVals,0.0]
        xVals = np.append(xVals,xMax)

    ax.plot(xVals,pVals,**priArgs)

    for ii,res in enumerate(allRes) :
        ## note these are simple lists now (not lists of single-element vectors)
        dataSet = [ rr[ dic ][ var ] for rr in res ]
        
        bKDE = bdyKDE(dataSet,kernelWidth,xMax=xMax,xMin=xMin,bdyMode='renorm')
        kPdf = np.array( [bKDE(x) for x in xValsKDE ])
        
        ## show the upper bound...
        if xMaxIsBound :
            xValsPlot = np.append(xValsKDE,xMax)
            kPdf = np.append(kPdf,0)
        
    #    ax.plot(xValsPlot,kPdf+(1+ii)*vOff,color=datCols[ii],
    #                      label='$t_{\\rm inf}='+'{t:d}$'.format(t=simTimeVals[ii]),
    #                      lw=2,)

        ax.fill_between(xValsPlot,kPdf+(1+ii)*vOff,(1+ii)*vOff,color=datCols[ii],
                          label='$t_{\\rm inf}='+'{t:d}$'.format(t=simTimeVals[ii]),
                          alpha=0.7,
                          #lw=2,)
                          )
       
    ### magic to print the legend with reversed order
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(reversed(handles), reversed(labels),
    #          loc='upper left',bbox_to_anchor=(1.03,1))
              
    ax.set_xlabel(labText)
    ax.set_ylabel('pdf')
    ax.set_xlim(0,0.22) ## hardcoded
    ax.set_xticks([0,0.1,0.2])  ## hardcoded
    ax.set_yticks([])

    plt.savefig(figPath+"winLateBKDE-offset.pdf")
    #plt.show(fig) ;
    plt.close()

## FIGS for posterior mean + std

def plotMeanStd(ax,dic,varList,legendLabs) : # ,figSize=(5,4),fileName=None,yLab=None):
    #fig = plt.figure(figsize=figSize)
    maxVal = 0.0

    for ii,lab in enumerate(varList) :
        meanList = []
        stdList = []
        for res in allRes :
            dat =  [ rr[dic][lab] for rr in res ]
            meanList.append( np.mean(dat) )
            stdList.append(   np.std(dat) )

            maxVal = np.maximum(maxVal,np.max(dat))

        ## special case to avoid overlapping data sets
        if lab == 'gammaIs3' : offset = 0.04
        elif lab == 'gammaIs2' : offset = -0.04
            
        else : offset = 0.0
        ax.errorbar(offset+np.array(simTimeVals),meanList,yerr=stdList,
                     fmt='o-',label=legendLabs[ii])

    ax.set_ylim(0,maxVal)

xTicks = list(range(7,11))

## FIG gammas

fig,axs = plt.subplots(1,1,figsize=(4.2,3.8))
plt.subplots_adjust(left=0.08,right=0.63,bottom=0.18,top=0.95)
ax = axs

dic = 'params_dict'
varList = ['gammaE','gammaA','gammaIs1','gammaIs2','gammaIs3']
legendLabs = ['$\\gamma_{\\rm E}$','$\\gamma_{\\rm A}$',
              '$\\gamma_{1}$','$\\gamma_{2}$','$\\gamma_{3}$']
plotMeanStd(ax,dic,varList,legendLabs)

ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0))
ax.set_xlabel('$t_{\\rm inf}$')
ax.set_xticks(xTicks)

plt.savefig(figPath+'winGamma.pdf')
plt.close()

## FIG lockdown time etc

fig,axs = plt.subplots(1,1,figsize=(4.2,3.8))
plt.subplots_adjust(left=0.08,right=0.56,bottom=0.18,top=0.95)
ax = axs

dic = 'control_params_dict'
varList = ['loc','width']
legendLabs = ['$t_{\\rm lock}$','$W_{\\rm lock}$']
plotMeanStd(ax,dic,varList,legendLabs)

ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.0))
ax.set_xlabel('$t_{\\rm inf}$')
ax.set_xticks(xTicks)

plt.savefig(figPath+'winLock.pdf')
plt.close()

figSize = (6,4)
fig = plt.figure(figsize=figSize)
maxVal = 0.0


## FIG : beta and aF

## (for two separate figs)
#fig,axs = plt.subplots(1,1,figsize=(4.2,3.8))
#plt.subplots_adjust(left=0.15,right=0.66,bottom=0.18,top=0.95)
#ax = axs

## (for one fig with two panels)
fig,axs = plt.subplots(1,2,figsize=(2*4.2,4.0))
plt.subplots_adjust(left=0.08,right=0.86,bottom=0.21,top=0.95,wspace=0.4)
ax = axs[0]

colMap = matplotlib.cm.rainbow(np.linspace(0.95, 0.0, numCohorts))

dic = 'params_dict'
lab = 'beta'
maxVal = 0.0
for ii in reversed(list(range(numCohorts))) :
    meanList = []
    stdList = []
    for res in allRes :
        dat =  [ rr[dic][lab][ii] for rr in res ]
        meanList.append( np.mean(dat) )
        stdList.append(   np.std(dat) )

        maxVal = np.maximum(maxVal,np.max(dat))

    ageLab = cohLabs[ii]
    ax.errorbar(np.array(simTimeVals),meanList,yerr=stdList,
                 fmt='o-',label=ageLab,
                 color=colMap[numCohorts-1-ii])

ax.set_ylim(0,maxVal)  ## hardcoded

#ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.08),prop={'size':12})
ax.set_xlabel('$t_{\\rm inf}$')
ax.set_ylabel('$\\beta_i$')
ax.set_xticks(xTicks)

#plt.savefig(figPath+'winBeta.pdf')
#plt.close()

## single fig
#fig,axs = plt.subplots(1,1,figsize=(4.2,3.8))
#plt.subplots_adjust(left=0.18,right=0.69,bottom=0.18,top=0.95)
#ax = axs

## second panel
ax = axs[1]

dic = 'control_params_dict'
lab = 'aO_f'
maxVal = 0.0
for ii in reversed(list(range(numCohorts))) :
    meanList = []
    stdList = []
    for res in allRes :
        dat =  [ rr[dic][lab][ii] for rr in res ]
        meanList.append( np.mean(dat) )
        stdList.append(   np.std(dat) )

        maxVal = np.maximum(maxVal,np.max(dat))

    ageLab = cohLabs[ii]
    ax.errorbar(np.array(simTimeVals),meanList,yerr=stdList,
                 fmt='o-',label=ageLab,
                 color=colMap[numCohorts-1-ii])

ax.set_ylim(0,0.5)  ## hardcoded

#ax.legend(loc='upper left',bbox_to_anchor=(1.02,1.08),prop={'size':12})
ax.set_xlabel('$t_{\\rm inf}$')
ax.set_ylabel('$a^{\\rm F}_i$')
ax.set_xticks(xTicks)



##-----------------------plot colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size="5%", pad=0.11,)

colMap = matplotlib.cm.rainbow(np.linspace(0.0, 0.95, numCohorts))
cma = matplotlib.colors.ListedColormap(colMap, name='from_list', N=None)
msm = matplotlib.cm.ScalarMappable(cmap=cma) 
msm.set_array(numCohorts) 

cb=plt.colorbar(msm, cax=cax)#. pad=0.1)
cb.set_ticks([14.5,15.5, 16.5, 17.5])
cb.set_ticklabels(['0-4','25-29','50-54','75+'])
cb.ax.tick_params(labelsize=16)
cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation='horizontal')
##-----------------------
#plt.savefig(figPath+'winAf.pdf')
plt.savefig(figPath+'winBeta_aF.pdf')

plt.close()
