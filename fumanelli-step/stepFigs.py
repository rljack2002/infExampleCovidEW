#!/usr/bin/env python
# coding: utf-8

## python script for making Figs
## step-like-NPI
## note we run deterministic dynamics to get MAP trajectory, requires stable pyross...

## output  in ../finalFigs
### beta_mcmc.pdf : inferred beta
### stepForeAll.pdf : forecast with different CM (see also stepForeSep.pdf)
### ageMAPandData.pdf : MAP trajectory, deaths by cohort, compared with data
### infLatentDet.pdf : posterior determinstic samples for latent compartments
### infSampAll.pdf : conditional nowcast of latent compartments
### FIM.pdf + FIM_soft.pdf + FIM_inf.pdf : fisher information matrix figs
### evi.pdf : evidence comparison (evi2.pdf is defunct)
### aO_mcmc.pdf : inf a-params for NPI
### ICinf.pdf : inf initial condition params
### otherInf.pdf : inf other params (lock time, lock width, nu_L)

## option to skip latent compartment figs, to save time
## (the conditional sampling takes 15 seconds or so...)
doLatent = True

## do we diagonalise the FIM or do we load the spectrum?
## if we diagonalise then we have to
##   export OMP_NUM_THREADS=1
## to ensure robustness of the result (due to hybridisation of small evals)
diagFIM = False

import numpy as np
from   matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm
import time

import pyross
import pickle
import pprint

import scipy.stats
import scipy.linalg

from ew_fns import *
import expt_params_local
import model_local

## filename root for stored results
pikFileRoot = "ewMod"

## where to put figs
figPath = "../finalFigs/"

## time unit is one week
daysPerWeek = 7.0

#print(scipy.linalg.__file__)
#print(np.__file__)

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

## cohort age ranges
cohRanges = [ [x,x+4] for x in range(0,75,5) ]
cohLabs = ["{l:d}-{u:d}".format(l=low,u=up) for [low,up] in cohRanges ]
cohLabs.append("75+")

for pp in ['left','right','top','bottom','wspace','hspace']:
  print('subP default',pp,plt.rcParams['figure.subplot.'+pp])

## we quote figsize numbers as if they were cm (in fact they are inches)
## this means 20 is ok as default font
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'serif'
plt.rc('text', usetex=True)

## model variants in these directories
#CMnames = ['CM-Fum','CM-Prem','CM-mix']
CMnames = ['$C^{\\rm F}$','$C^{\\rm P}$','$C^{\\rm M}$']
cmDirs = ['../fumanelli-step/','../prem-step/','../mix-step/']

## LOAD data
saveTrajList = []
resList = []
mcParamList = []

for cmDir in cmDirs :
    ipFile = cmDir + pikFileRoot + "-traj_mcmc.pik"
    print('ipf',ipFile)
    with open(ipFile, 'rb') as f: 
        [model_spec,saveTraj,N,numCohorts,deathCumulativeDat] = pickle.load(f)
    saveTrajList.append(saveTraj)
    
    ipFile = cmDir + pikFileRoot + "-inf.pik"
    print('ipf',ipFile)
    with open(ipFile, 'rb') as f: 
        [infResult,tt] = pickle.load(f)
    resList.append(infResult)
    
    ipFile = cmDir + pikFileRoot + "-result_mcmc.pik"
    print('ipf',ipFile)
    with open(ipFile, 'rb') as f:
        [result_mcmc] = pickle.load(f)
    mcParamList.append(result_mcmc)
    
print(len(saveTrajList))
print(len(resList))
print(len(mcParamList))

ezDir = '../fumanelli-ez/'

ipFile = ezDir + pikFileRoot + "-result_mcmc.pik"
print('ipf',ipFile)
with open(ipFile, 'rb') as f:
    [result_mcmc_ez] = pickle.load(f)

ipFile = ezDir + pikFileRoot + "-inf.pik"
print('ipf',ipFile)
with open(ipFile, 'rb') as f:
    [infResult_ez,tt] = pickle.load(f)

CMnamesAll = CMnames.copy()
CMnamesAll += ['Fum-ez']

colsCM = ['C3','darkorchid','limegreen' , 'C3']  ## for contact matrices
markCM = [ '*', 'D' , 's', 'o']                 ## for contact matrices
msCM = [12,6,6,8]                          ## markersizes

## FIG : inferred beta

eta = daysPerWeek  ##  for beta rescaling

## TWO panels

fig,axs = plt.subplots(1,2,figsize=(8.5,3.5),sharey=True)
plt.subplots_adjust(top=0.97,bottom=0.15,wspace=0.27,left=0.08,right=0.97)

## RIGHT PANEL first
ax = axs[1]

## MAP
maxB = 0.0
for ffi,rr in enumerate(resList) :
    ax.plot([xx/eta for xx in rr['params_dict']['beta']],ms=msCM[ffi],marker=markCM[ffi],
                             color=colsCM[ffi],label=CMnames[ffi],linestyle='none')
    maxB = np.maximum( maxB, 1.0/eta * np.max(rr['params_dict']['beta']) )

ax.set_ylim(0,maxB*1.2)
ax.legend(frameon=False,bbox_to_anchor=(-0.03, 1.03),
                        loc='upper left',
                        handlelength=1.0)

# which cohorts to label on x-axis
tickList = [0,4,8,12,15]
ax.set_xticks([ii for ii in tickList] )
ax.set_xticklabels( [ cohLabs[ii] for ii in tickList ] )
ax.set_xlabel('age')
ax.tick_params(labelleft=True)
ax.set_ylabel('$\\beta_i$ (MAP)')

## LEFT PANEL
ax = axs[0]

infResult = resList[0]        ## MAP
result_mcmc = mcParamList[0]  ## posterior MCMC

mapArgs =  { 'marker' : '*' , 'ms' : 12, 'color' : 'C3' , 'linestyle' : 'none', 'label' : 'MAP' }
priArgs =  { 'marker' : 'X' , 'ms' : 8 , 'color' : 'C2' , 'linestyle' : 'none', 'label' : 'prior mean'}
postArgs = { 'marker' : 'o' , 'ms' : 7 , 'color' : 'C0' , 'linestyle' : 'none' ,'label' : 'posterior mean' }

dictLab = 'params_dict'
datLab = 'beta'
allData = []
cList = list(range(numCohorts))
for c in cList :
    allData.append( np.array( [rr[dictLab][datLab][c]/eta for rr in result_mcmc ] ))

## MCMC percentiles
datMean = np.array( [np.mean(ss) for ss in allData ] )
datMedian = np.array( [np.percentile(ss,50) for ss in allData ] )
datLower = np.array( [np.percentile(ss,5) for ss in allData ] )
datUpper = np.array( [np.percentile(ss,95) for ss in allData ] )

## post mean
ax.plot(cList,datMean,**postArgs,alpha=1.0)
## prior
ax.plot(cList,1.0/eta * priorsAll['beta']['mean'],**priArgs)
## MAP
ax.plot(cList,1.0/eta * infResult['params_dict']['beta'],**mapArgs)
## post range
ax.fill_between(cList,datLower,datUpper,alpha=0.3)

## ticks etc
ax.set_xticks([ii for ii in tickList] + [ numCohorts-1 ] )
ax.set_xticklabels( [ cohLabs[ii] for ii in tickList ] + [ cohLabs[-1] ] )
ax.set_xlabel('age')
ax.tick_params(labelleft=True)
ax.set_ylabel('$\\beta_i$')
ax.legend(frameon=False,bbox_to_anchor=(-0.03, 1.03),
                        loc='upper left',
                        handlelength=1.0,
                        handletextpad=0.2)

plt.savefig(figPath+'beta_mcmc.pdf',bbox_inches='tight')
#plt.show()
plt.close()



## FIG : weekly deaths and forecast, all step-like models


## for weekly deaths
def diffs(dd) :
    op=[]
    for ii,di in enumerate(dd[1:]) : op.append( di-dd[ii] )
    return op


runTime = 10 
simTime = 7

nSel = 20  ## how many traj for each CM

fig,axs = plt.subplots(1,1,figsize=(4.0,4.8))
ax = axs
tVals = np.linspace(1,runTime,runTime)

for ii in range(nSel) :
    for ffi,saveTraj in enumerate(saveTrajList) :
        mytraj = saveTraj[-ii]
        cc=model_spec['classes'].index('Im')
        dMod = N * np.sum(mytraj[:,(cc*numCohorts):((cc+1)*numCohorts)],axis=1)
        #if ii == 0 : lab=CMnames[ffi] (old version of labelling, for legend)
        #else : lab=None
        lab=None
        ax.plot(tVals,diffs(dMod), '-', label=lab , color=colsCM[ffi] , alpha=0.4, linewidth=1 )
        ## for legend (new version ensures alpha=1 in legend)
        if ii == 0 : ax.plot([],[],color=colsCM[ffi] , label=CMnames[ffi], alpha=1.0, linewidth=1)
plt.xlabel('time (weeks)')
plt.ylabel('weekly deaths')

ax.axvspan(1, simTime,alpha=0.2, color='silver',
           label='inf window')

ax.set_xticks( [xx for xx in range(0,11,2) ])

dObs = np.sum(deathCumulativeDat[0:numCohorts,:runTime+1].transpose(), axis=1)
ax.plot(tVals,diffs(dObs), 'o', label='data' ,color='orange', ms=5 )

plt.legend(bbox_to_anchor=(1, 1.0) ) # ,frameon=False)
plt.savefig(figPath+'stepForeAll.pdf',bbox_inches='tight')
#plt.show(fig)
plt.close()

## alternative FIG : separate panels

fig,axs = plt.subplots(1,3,figsize=(3*6.5,4.8))
ax = axs[0]
tVals = np.linspace(1,runTime,runTime)

for ii in range(nSel) :
    for ffi,saveTraj in enumerate(saveTrajList) :
        mytraj = saveTraj[-ii]
        cc=model_spec['classes'].index('Im')
        dMod = N * np.sum(mytraj[:,(cc*numCohorts):((cc+1)*numCohorts)],axis=1)
        #print(dMod)
        if ii == 0 : lab=CMnames[ffi]
        else : lab=None
        axs[ffi].plot(tVals,diffs(dMod), '-', label=lab , color=colsCM[ffi] , alpha=0.4, linewidth=1 )
plt.xlabel('time')
plt.ylabel('weekly deaths')

ax.axvspan(1, simTime,alpha=0.2, color='silver',
           label='inference window')

dObs = np.sum(deathCumulativeDat[0:numCohorts,:runTime+1].transpose(), axis=1)
for ax in axs :
    ax.plot(tVals,diffs(dObs), 'o', label='data' ,color='orange', ms=5 )

plt.legend(bbox_to_anchor=(1, 1.0) ) # ,frameon=False)
plt.savefig(figPath+'stepForeSep.pdf',bbox_inches='tight')
#plt.show(fig)
plt.close()


## ** from this point we use only fumanelli data

infResult = resList[0]         ## MAP for fumanelli
result_mcmc = mcParamList[0]   ## posterior for fumanelli


## FIG : "rainbow"  with deaths by cohort


epiParamsMAP = infResult['params_dict']
conParamsMAP = infResult['control_params_dict']
x0_MAP = infResult['x0']

CM_MAP = contactBasis.intervention_custom_temporal( interventionFn,
                                                    **conParamsMAP)

estimator.set_params(epiParamsMAP)
estimator.set_contact_matrix(CM_MAP)
trajMAP = estimator.integrate( x0_MAP, exptParams['timeZero'], simTime, simTime+1)

fig,axs = plt.subplots(1,2,figsize=(10,4.5),sharey=True)
plt.subplots_adjust(left=0.09,right=0.86,wspace=0.18,bottom=0.18,top=0.9)
## ... color map choices ...
colMap = matplotlib.cm.rainbow(np.linspace(0.95, 0.0, numCohorts))
#colMap = matplotlib.cm.cool(np.linspace(1, 0, numCohorts))
#colMap = matplotlib.cm.hsv(np.linspace(1, 0, numCohorts))
#colMap = matplotlib.cm.tab20b(np.linspace(1, 0, numCohorts))

## RIGHT panel is model

stdFont = 20

ax = axs[1]
ax.set_title('MAP model',fontsize=stdFont)
mSize = 3
minY = 0.3
maxY = 1.0
indClass = model_spec['classes'].index('Im')
ax.set_yscale('log')
ax.set_xlabel('time (weeks)')
for ii,coh in enumerate(reversed(list(range(numCohorts)))) :
    plotX = np.linspace(0,simTime,simTime+1).tolist()
    plotY = (N*trajMAP[:,coh+indClass*numCohorts]).tolist()
        
    ## clear out any initial elements below 0.1
    while len(plotY)>0 and plotY[0] < 0.1 : plotX.pop(0) ; plotY.pop(0)
    
    ax.plot( plotX,plotY,'o-',label=cohLabs[coh],ms=mSize,color=colMap[numCohorts-1-coh] )
    maxY = np.maximum( maxY, np.max(N*trajMAP[:,coh+indClass*numCohorts]))
#ax.legend(fontsize=8,bbox_to_anchor=(1, 1.0))
maxY *= 1.6
ax.set_ylim(bottom=minY,top=maxY)
ax.tick_params(labelleft=True)

#ax.legend(fontsize=14,bbox_to_anchor=(1, 1.15))

#plt.show() ; plt.close()

## LEFT panel is data

ax = axs[0]
ax.set_title('data',fontsize=stdFont)
ax.set_xlabel('time (weeks)')
ax.set_ylabel('cumulative deaths')

indClass = model_spec['classes'].index('Im')
ax.set_yscale('log')
for coh in range(numCohorts-1,-1,-1) :
    
    plotX = np.linspace(0,simTime,simTime+1).tolist()
    plotY = (N*obsDeath[:,coh]).tolist()
    
    ## clear out any initial elements below 0.1
    while len(plotY)>0 and plotY[0] < 0.1 : plotX.pop(0) ; plotY.pop(0)
    
    ax.plot( plotX,plotY,'o-',label=cohLabs[coh],ms=mSize,color=colMap[numCohorts-1-coh] )
    
    #ax.plot( ,'o-',label=cohLabs[coh],ms=mSize,color=colMap[ii] )
## keep the same as other panel
ax.set_ylim(bottom=minY,top=maxY)

ax.set_yticks([10**i for i in range(5)])
## magic to force inclusion of minor ticks
bigNum = 12  ## arbitrary big number
tickSetup = matplotlib.ticker.LogLocator(base=10.0,subs=[0.1*(1+x) for x in range(9)],numticks=bigNum)
ax.yaxis.set_minor_locator(tickSetup)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


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

plt.savefig(figPath+'ageMAPandData.pdf')
plt.close()


if not doLatent :
    print('skipping latent figs')

else :

    ## FIG : latent compartments (deterministic), parameters from posterior
    ## same for posterior

    colsClass = {'E' : 'C1' , 'A':'C2', 'Is1':'C3','Is2':'C4','Is3':'C5'}
    legLab = { 'E' : 'E' , 'A' : 'A' , 'Is1' : 'I$^{(1)}$', 'Is2' : 'I$^{(2)}$', 'Is3' : 'I$^{(3)}$' }
    fig,ax = plt.subplots(1,1,figsize=(5.5,4.8))

    nPlot = 40
    for iir,rr in enumerate(result_mcmc[-nPlot:]) :
        CM_samp = contactBasis.intervention_custom_temporal( interventionFn,
                                                            **rr['control_params_dict'])

        estimator.set_params(rr['params_dict'])
        estimator.set_contact_matrix(CM_samp)

        fineRes = 7 ## number of points to plot per week
        trajMAPfine = estimator.integrate( rr['x0'], exptParams['timeZero'], simTime, (simTime+1)*fineRes )

        yesPlot = model_spec['classes'].copy()
        yesPlot.remove('S')
        yesPlot.remove('Im')
        plt.yscale('log')
        for lab in yesPlot :
            indClass = model_spec['classes'].index(lab)
            tVals = np.linspace(0,simTime,(simTime+1)*fineRes)
            totClass = np.sum(trajMAPfine[:,indClass*numCohorts:(indClass+1)*numCohorts],axis=1)
            if iir == 0 :
                ## fake data series with alpha=1, used only for legend
                plt.plot( [],[],'-',lw=1,label=legLab[lab],alpha=1.0,color=colsClass[lab])
            plotLab = None
            plt.plot( tVals,N * totClass,'-',lw=1,label=plotLab,alpha=0.3,color=colsClass[lab])

        
        plt.ylabel('latent class population')
        plt.xlabel('time (weeks)')
        
    plt.legend(fontsize=stdFont,bbox_to_anchor=(1, 1.0),handlelength=1.0)
    plt.savefig(figPath+"infLatentDet.pdf",bbox_inches='tight')
    plt.close()


    ## FIG : latent compartments conditional on data

    ## get some data

    estimator.set_params(epiParamsMAP)
    CM_MAP = contactBasis.intervention_custom_temporal( interventionFn,
                                                        **conParamsMAP)
    estimator.set_contact_matrix(CM_MAP)

    nsamples = 100
    print('... sampling {n:d} latent trajectories'.format(n=nsamples))
    sampleTime = simTime

    elapsedSamp = -time.time()
    trajs = estimator.sample_trajs(obsDeath, fltrDeath, simTime, infResult, nsamples, contactMatrix=CM_MAP )
    elapsedSamp += time.time()
    print('** time',elapsedSamp,'secs')

    tValsMAP = np.linspace(0,simTime,simTime+1)
    tValsSample = np.linspace(1,simTime,simTime)

    ## TWO panels

    fig,axs = plt.subplots(1,2,figsize=(8.5,4.0))
    plt.subplots_adjust(left=0.11,right=0.84,wspace=0.33,top=0.98,bottom=0.19)
    ax = axs[0]

    yesPlot = model_spec['classes'].copy()
    yesPlot.remove('S')
    yesPlot.remove('Im')

    includeInitAsPoints = True

    ax.set_xticks( list(range(0,7,2)) )

    ax.set_xlabel('time (weeks)')
    ax.set_ylabel('population (all cohorts)'.format(n=nsamples))
    ax.set_yscale('log')
    cols = [ "C{n:d}".format(n=x) for x in range(8) ]
    for ii,lab in enumerate(yesPlot) :
        indClass = model_spec['classes'].index(lab)
        totClassMAP = np.sum(trajMAP[:,indClass*numCohorts:(indClass+1)*numCohorts],axis=1)

        #plt.plot( tValsMAP, N * totClass,'o-',lw=2,ms=3,label=lab)
        for jj,traj in enumerate(trajs):
            if jj == 0 : pLab = legLab[lab]
            else : pLab = None
            totClass = np.sum(traj[:,indClass*numCohorts:(indClass+1)*numCohorts],axis=1)
            ax.plot( tValsSample, N * totClass,'-',lw=1,ms=3,color=cols[ii],label=pLab)
            
        if includeInitAsPoints :
            ax.plot( [0], [N*totClassMAP[0]],'o',lw=1,ms=3,color=cols[ii],label=pLab)

    abLab = 0.04
    abLabV = 0.95
    ax.text(abLab, abLabV, '(a)', transform=ax.transAxes, va='top')

    ax = axs[1]

    minusCoh = 11  ## which cohort? 1 is oldest (ie 75+) 16 is youngest (0-4)

    ax.set_xticks( list(range(0,7,2)) )

    ax.set_xlabel('time (weeks)')
    ax.set_ylabel('population (age '+cohLabs[-minusCoh] +')' )
    ax.set_yscale('log')
    cols = [ "C{n:d}".format(n=x) for x in range(8) ]
    for ii,lab in enumerate(yesPlot) :
        indClass = model_spec['classes'].index(lab)
        totClassMAP = trajMAP[:,-minusCoh+(indClass+1)*numCohorts]

        #plt.plot( tValsMAP, N * totClass,'o-',lw=2,ms=3,label=lab)
        for jj,traj in enumerate(trajs):
            if jj == 0 : pLab = legLab[lab]
            else : pLab = None
            totClass = traj[:,-minusCoh+(indClass+1)*numCohorts]
            tVals = tValsSample
                
            ax.plot( tVals, N * totClass,'-',lw=1,ms=3,color=cols[ii],label=pLab)
            
        if includeInitAsPoints :
            ax.plot( [0], [N*totClassMAP[0]],'o',lw=1,ms=3,color=cols[ii],label=pLab)

    ax.text(abLab, abLabV, '(b)', transform=ax.transAxes, va='top')

    #plt.plot(N*np.sum(obsDeath,axis=1),'X',label='data')
    plt.legend(fontsize=stdFont,bbox_to_anchor=(1, 1.0),handlelength=1.0)
    plt.savefig(figPath+'infSampAll.pdf')
    #plt.show() ;
    plt.close()

## posterior FIGS

## used for prior pdfs
(likFun,priFun,dimFlat) = pyross.evidence.latent_get_parameters(estimator,
                                    obsDeath, fltrDeath, simTime,
                                    priorsAll,
                                    initPriorsLinMode,
                                    generator=contactBasis,
                                    intervention_fun=interventionFn,
                                    tangent=False,
                                  )

## for consistency for plotting symbols etc...
##    note mapArgs,priArgs,postArgs were defined above

## violins...
def vioPlotGen(ax,dictLab,addLabs,ylab=None,fileName=None,xlabs=None) :
    buildBox = [ [dictLab,lab] for lab in addLabs ]
    boxData = [ [ rr[dLab][lab] for rr in result_mcmc ] for [dLab,lab] in buildBox ]
    boxYmax = 1.1 * np.max(np.array(boxData))

    if xlabs == None : xlabs = addLabs
    
    #plt.boxplot(boxData,labels=addLabs)
    ax.violinplot(boxData, showextrema=True,
                   showmedians=True ,
                   #quantiles=[[0.25,0.75]]*len(boxData)
                  )#,labels=addLabs)
    ax.set_xticks(list(range(1,1+len(addLabs))))
    ax.set_xticklabels(xlabs)
    
    nLabs=len(addLabs)
    ax.plot([ii for ii in range(1,nLabs+1)],
             [infResult[dictLab][lab] for lab in addLabs],**mapArgs)
    ax.plot([ii for ii in range(1,nLabs+1)],
             [priorsAll[lab]['mean'] for lab in addLabs],**priArgs)

    ax.set_ylim(0,boxYmax)
    ax.set_ylabel(ylab)
    ax.legend(frameon=False,handlelength=1.0)

## posterior for gammas

fig,ax=plt.subplots(1,1,figsize=(6,4.8))
plt.subplots_adjust(bottom=0.15,top=0.98,right=0.99,left=0.15)

addLabs = ['gammaE','gammaA','gammaIs1','gammaIs2','gammaIs3']
plotLabs = ['$\\gamma_{\\rm E}$','$\\gamma_{\\rm A}$','$\\gamma_{1}$','$\\gamma_{2}$','$\\gamma_{3}$']
vioPlotGen(ax,'params_dict',addLabs,ylab='rate (per week)',xlabs=plotLabs)

plt.savefig(figPath+"gammaInf.pdf")
plt.close()

## posterior for aO

yesTick = [0,4,8,12,15]   ## which cohorts to label

fig,ax=plt.subplots(1,1,figsize=(6,4.8))
plt.subplots_adjust(bottom=0.15,top=0.98,right=0.99,left=0.15)
## collate the data
dictLab = 'control_params_dict'
datLab = 'aO_f'
allData = []
cList = list(range(numCohorts))
for c in cList :
    allData.append( np.array( [rr[dictLab][datLab][c] for rr in result_mcmc ] ))

## percentiles etc
datMean = np.array( [np.mean(ss) for ss in allData ] )
datMedian = np.array( [np.percentile(ss,50) for ss in allData ] )
datLower = np.array( [np.percentile(ss,5) for ss in allData ] )
datUpper = np.array( [np.percentile(ss,95) for ss in allData ] )

# mean
ax.plot(cList,datMean,**postArgs)
# prior
ax.plot(cList,priorsAll['aO_f']['mean'],**priArgs)
# MAP
ax.plot(cList,infResult['control_params_dict']['aO_f'],**mapArgs)
# shading
ax.fill_between(cList,datLower,datUpper,alpha=0.3)

## manual upper bound for axis
yMax = np.max(datUpper)*1.5
ax.set_ylim(top=yMax,bottom=0)

ax.set_xlabel('age')
ax.set_xticks(yesTick)
ax.set_xticklabels([cohLabs[ii] for ii in yesTick])
ax.set_ylabel('$a^{\\rm F}_i$') # (shading 5th to 95th \%)')
ax.legend(frameon=False,loc='upper left',fontsize=stdFont,handlelength=1.0)
plt.savefig(figPath+'aO_mcmc.pdf')
#plt.show() ;
plt.close()

## posterior for lock params etc

nRow = 1
nCol = 3
figSize = (nCol*3.2,nRow*2.5)  ## ballpark

datArgs = { 'label' : 'post' , 'lw' : 2 , 'color' : 'C0' }
priArgs = { 'label' : 'prior', 'lw' : 2 , 'color' : 'C1' }

toPlot = []
toPlot.append(['control_params_dict','loc','$t_{\\rm lock}$ (weeks)'])
toPlot.append(['control_params_dict','width','$W_{\\rm lock}$ (weeks)'])
#toPlot.append(['control_params_dict','easeFrac','lockdown easing factor'])
toPlot.append(['params_dict','betaLateFactor','$\\nu_{\\rm L}$'])

fig,axs = plt.subplots(nRow,nCol,figsize=figSize)
plt.subplots_adjust(left=0.07,right=0.97,bottom=0.28,wspace=0.35,top=0.99)

panelID = 0

while panelID < len(toPlot) and panelID < nRow*nCol :
    if nRow == 1 :
        if nCol == 1 : ax=axs
        else : ax = axs[panelID]
    else :
        row = int(panelID/nRow)
        ax = axs[row][panelID-(nCol*row)]

    [dic,var,labText] = toPlot[panelID]
    dataSet = [ rr[ dic ][ var ] for rr in result_mcmc ]
    ax.hist(dataSet,histtype='step',density=True,**datArgs)
    ax.set_xlabel(labText)
    if panelID % nCol == 0 :
        ax.set_ylabel('pdf')

    xMin = 0.0
    xMaxData = np.max(dataSet)
    xMaxPlot = 1.2*xMaxData
    
    if xMaxPlot > priorsAll[var]['bounds'][1] :
        xMaxPrior = priorsAll[var]['bounds'][1]
    else :
        xMaxPrior = xMaxPlot
        
    xVals = np.linspace(xMin,xMaxPrior,100)
    
    ## magic to work out the index of this param in flat_params
    jj = infResult['param_keys'].index(var)
    xInd = infResult['param_guess_range'][jj]
    
    pVals = []
    for xx in xVals :
        flatP = np.zeros( dimFlat )
        flatP[xInd] = xx
        pdfAll = np.exp( priFun.logpdf(flatP) )
        pVals.append( pdfAll[xInd] )

    ax.plot(xVals,pVals,**priArgs)

    ax.set_xlim(0,xMaxPlot)
    
    if panelID == 0 :
        ax.legend(frameon=False,handlelength=1.0,
                  loc='upper left',bbox_to_anchor=(0.03, 1.03))
    
    panelID += 1
    
plt.savefig(figPath+"otherInf.pdf")
plt.close()

## posterior for initial condition params

nRow = 2
nCol = 3
figSize = (nCol*3.3,nRow*3)  ## ballpark

fig,axs = plt.subplots(nRow,nCol,figsize=figSize)
plt.subplots_adjust(left=0.12,right=0.97,bottom=0.17,wspace=0.51,top=0.92,hspace=0.4)

nPanel = nRow*nCol

labs = ['$\\kappa\\Omega$','E$_0$','A$_0$','I$^{(1)}_0$','I$^{(2)}_0$','I$^{(3)}_0$']

for panelID in range(nPanel) :

    if nRow == 1 :
        if nCol == 1 : ax=axs
        else : ax = axs[panelID]
    else :
        row = int(panelID/nCol)
        ax = axs[row][panelID-(nCol*row)]

    paramID = panelID-nPanel  ## negative array index for initial condition param
    
    dataSet = [ N*rr[ 'flat_params' ][ paramID ] for rr in result_mcmc ]
    ax.hist(dataSet,histtype='step',density=True,**datArgs)
        
    xMin = 1.0/N
    xMax = 1.1*np.max(dataSet)/N
    
    xVals = np.linspace(xMin,xMax,100)
    #ax.plot(N*xVals,[scipy.stats.lognorm.pdf(xx,s,scale=scale)/N for xx in xVals],**priArgs)

    pVals = []
    for xx in xVals :
        flatP = np.zeros( dimFlat )
        flatP[paramID] = xx
        pdfAll = np.exp( priFun.logpdf(flatP) )
        pVals.append( pdfAll[paramID] )

    ax.plot(N*xVals,np.array(pVals)/N,**priArgs)
    
    #print(N*xVals,[scipy.stats.lognorm.pdf(xx,s,scale=scale)/N for xx in xVals])
    
    ax.set_xlabel(labs[panelID])
    if panelID % nCol == 0 :
        ax.set_ylabel('pdf')
    #if panelID == 0 : ax.legend()


plt.savefig(figPath+"ICinf.pdf")
#plt.show(fig) ;
plt.close()

#print(infResult)

### figures with FIM info (disabled pending notebook upd)

infResult = resList[0]

FIM = np.load(pikFileRoot+'-FIM.npy')
print(FIM[:3,:3])
No = np.arange(0, len(infResult['flat_params']))

evals,evecs = 0,0

if diagFIM :
    print('diagonalising FIM')
    evals, evecs = scipy.linalg.eigh(FIM)

    opFile = pikFileRoot + "-FIM-evec.pik"
    print('opf',opFile)
    with open(opFile, 'wb') as f:
        pickle.dump([evals, evecs],f)

else :
    print('loading FIM spectrum')

    ipFile = pikFileRoot + "-FIM-evec.pik"
    with open(ipFile, 'rb') as f:
        [evals, evecs] = pickle.load(f)

print('evals',evals)

sens = np.sqrt(np.diagonal(FIM))*infResult['flat_params']

## indices
betas_ = np.s_[0:16]
#betaL_ = 16
gammas_ = np.s_[16:22] # include betaLate with gammas
aF_ = np.s_[22:38]
lock_ = np.s_[38:40]
inits_ = np.s_[40:46]

fig,axs = plt.subplots(2,2,figsize=(8.5,6),sharey=True,gridspec_kw={'width_ratios': [1,1.5]})
plt.subplots_adjust(wspace=0.35,right=0.97,left=0.11,bottom=0.13,top=0.97,hspace=0.5)

plotLabs_g = ['$\\nu_{L}$',
            '$\\gamma_{\\rm E}$','$\\gamma_{\\rm A}$',
            '$\\gamma_{1}$','$\\gamma_{2}$','$\\gamma_{3}$']

row,col = 0,0
ax = axs[row,col]

ax.set_yscale('log')
ax.set_ylim(0.2, 2000)
ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=15))
bigNum = 12  ## arbitrary big number
tickSetup = matplotlib.ticker.LogLocator(base=10.0,
                                         subs=[0.1*(1+x) for x in range(9)],
                                         numticks=bigNum)
ax.yaxis.set_minor_locator(tickSetup)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax.set_xticks(No[gammas_])
ax.set_xticklabels(plotLabs_g)
ax.set_ylabel('sensitivity $s_a$', labelpad=6)
ax.plot(No[gammas_], sens[gammas_], 'X', c='blue', markersize=7)
ax.axhline(y=1e2, color='red', alpha=0.3, lw=2)

row,col = 1,0
ax = axs[row,col]

plotLabs = ['$t_{\\rm lock}$',
           '$W_{\\rm lock}$']

ax.set_xticks(No[lock_])
ax.set_xticklabels(plotLabs)
ax.set_ylabel('sensitivity $s_a$', labelpad=6)

ax.plot(No[lock_], sens[lock_], 'X', c='blue', markersize=7)
ax.axhline(y=1e2, color='red', alpha=0.3, lw=2)

ax.set_xlim(No[lock_][0]-0.5,No[lock_][-1]+0.5)


## sensitivity for betas
yesTick = [0,5,10,15] # [0,4,8,12,15]

row,col = 0,1
ax = axs[row,col]

ax.set_ylabel('sensitivity for $\\beta_{i}$', labelpad=6)
ax.tick_params(labelleft=True)

ax.set_xticks(yesTick)
ax.set_xticklabels([cohLabs[ii] for ii in yesTick])

ax.plot(No[betas_], sens[betas_], 'X', c='blue', markersize=5)
ax.axhline(y=1e2, color='red', alpha=0.3, lw=2)
#ax.ylim(bottom=1, top=300)
ax.set_xlabel('age')
#plt.show(); plt.close()

## sensitivity for aF

row,col = 1,1
ax = axs[row,col]

ax.set_ylabel('sensitivity for $a^{\\rm F}_{i}$', labelpad=6)
ax.tick_params(labelleft=True)
ax.tick_params(labelbottom=True)

aTick = [ yy+No[aF_][0] for yy in yesTick ]

ax.set_xticks(aTick)
ax.set_xticklabels([cohLabs[ii] for ii in yesTick])

ax.plot(No[aF_], sens[aF_], 'X', c='blue', markersize=5)
ax.axhline(y=1e2, color='red', alpha=0.3, lw=2)
#ax.ylim(bottom=1, top=300)
ax.set_xlabel('age')
#plt.show(); plt.close()


plt.savefig(figPath+"FIM.pdf")

## sensitivity for inits
fig, ax = plt.subplots(1,1,figsize=(5, 3.8))
plt.subplots_adjust(left=0.18,right=0.97,bottom=0.15,top=0.97)

plotLabs = ['$\\kappa$',
            '${\\rm E}_{0}$',
            '${\\rm A}_{0}$',
            '${\\rm I}_{0}^{(1)}$',
            '${\\rm I}_{0}^{(2)}$',
            '${\\rm I}_{0}^{(3)}$']

ax.set_xticks(No[inits_])
ax.set_xticklabels(plotLabs)
ax.set_ylabel('sensitivity $s_a$',labelpad=6)

ax.semilogy(No[inits_], sens[inits_], 'X', c='blue', markersize=7)
ax.set_ylim(top=300.)
#ax.set_xlim(37.5, 40.5)
ax.axhline(y=1e2, color='red', alpha=0.3, lw=2)

plt.savefig(figPath+"FIM_inits.pdf")
plt.close()

## second eigenvalue = 3.8e-5
maps = infResult['flat_params']

evec_soft = evecs[:,1].copy()/maps

#print('maps\n',maps)
#print('evals\n',evals)
#print('evec\n',evec_soft)

sortOrder = np.flip( np.argsort(np.abs(evec_soft[:40])) )
print(sortOrder[:6])

softgA   = evec_soft[18]
softgI1  = evec_soft[19]
softgI2  = evec_soft[20]
softgI3  = evec_soft[21]
softaF   = evec_soft[28] ## for age 30-34

params_soft = [softgA,softgI1,softgI2,softgI3,softaF]

fig, ax = plt.subplots(1,1,figsize=(4.5,3.4))
plt.subplots_adjust(left=0.27,bottom=0.13,top=0.97,right=0.97)

plotLabs_soft = ['$\\gamma_{\\rm A}$',
                '$\\gamma_{1}$','$\\gamma_{2}$',
                '$\\gamma_{3}$',
                '$a^{F}_{30-34}$']

ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(plotLabs_soft)

#ax.set_yticks([])
ax.set_ylabel('$\\tilde{v}^{\\rm FIM}_a$')

for ii,pp in enumerate(params_soft) :
    ax.arrow( ii+1,0,0,pp,width=0.2*np.abs(pp),
             head_width=1.3*np.abs(pp), length_includes_head=True,
             head_length=0.3*np.abs(pp), ec='C0' )


ax.axhline(y=0, color='grey', alpha=0.3, lw=2, linestyle='dotted')

maxPos = np.max(params_soft)
maxModNeg = -np.min(params_soft)
maxMod = np.maximum(maxPos,maxModNeg)
maxMod *= 1.2

ax.set_xlim(0.5,len(params_soft)+0.5)
ax.set_ylim(-maxMod,maxMod)

plt.savefig(figPath+"FIM_soft.pdf")
plt.close()

#plt.show()
#plt.close()


## LIKELIHOODS and EVIDENCE

mapLik = []
for ii,infResult in enumerate(resList+[infResult_ez]) :
    mapLik.append(infResult['log_likelihood'])

for ii,result_mcmc in enumerate(mcParamList+[result_mcmc_ez]) :
    likVals = [ rr['log_likelihood'] for rr in result_mcmc ]
    priVals = [ rr['log_posterior']-rr['log_likelihood'] for rr in result_mcmc ]
    postVals = [ rr['log_posterior'] for rr in result_mcmc ]
    print(CMnamesAll[ii],'meanLik\t{lik:.1f} \tstd {likS:.1f}\t meanPost {post:.1f} meanPri {pri:.1f} mapLik {map:.1f}'.format(
                      lik =np.mean(likVals),
                      likS=np.std(likVals),
                      post=np.mean(postVals),
                      pri=np.mean(priVals),
                      map=mapLik[ii] ) )


## hardcoded ... should be loaded
evidenceVals = [ -316.7 , -314.4 , -321.7, -315.5 ]

postLikAll = [ np.mean( [ rr['log_likelihood'] for rr in result_mcmc ] )
                for ii,result_mcmc in enumerate(mcParamList+[result_mcmc_ez]) ]

postLikStdAll = [ np.std( [ rr['log_likelihood'] for rr in result_mcmc ]/np.sqrt(len(result_mcmc)) )
                for ii,result_mcmc in enumerate(mcParamList+[result_mcmc_ez]) ]


postPriAll = [ np.mean( [ rr['log_posterior'] - rr['log_likelihood'] for rr in result_mcmc ] )
                for ii,result_mcmc in enumerate(mcParamList) ]

postPriStdAll = [ np.std( [ rr['log_posterior'] - rr['log_likelihood'] for rr in result_mcmc ]/np.sqrt(len(result_mcmc)) )
                for ii,result_mcmc in enumerate(mcParamList) ]

print(postPriAll)
print(postPriStdAll)

CMnamesAllShort = ['$C^{\\rm F}$','$C^{\\rm P}$','$C^{\\rm M}$','$C^{\\rm F}$-ez']

## VERSION 1

fig,axs = plt.subplots(1,2,figsize=(8.5,3.5),sharey=False)
plt.subplots_adjust(wspace=0.45,right=0.9)

xTick = list(range(len(evidenceVals)))

ax = axs[0]

#ax.xaxis.tick_top()
ax.set_xticks(xTick)
ax.set_xticklabels(CMnamesAllShort)
ax.set_ylabel('$\\log Z$')
for ii,ee in enumerate(evidenceVals) :
     ax.plot(ii,ee,'s',color='black',linestyle='none')
#    ax.plot(ii,ee,marker=markCM[ii],ms=msCM[ii],color=colsCM[ii],
#                  label=CMnamesAll[ii],linestyle='none')

yRange = 13 ## same for both panels

yMin = -325
yMax = yMin+yRange
ax.set_ylim(yMin,yMax)

ax.set_xlim(-0.5,-0.5+len(evidenceVals))

ax = axs[1]
## unfortunately we have to put this axis label "by hand" since matplotlib does not like \mathbb
#ax.set_ylabel('${\\cal L}_{\\rm post}$')

yMin = -278
yMax = yMin+yRange
ax.set_ylim(yMin,yMax)

ax.set_xlim(-0.5,-0.5+len(evidenceVals))

for ii,ee in enumerate(postLikAll) :
    ax.plot(ii,ee,'o',color='C0',linestyle='none')
#    ax.plot(ii,ee,marker=markCM[ii],ms=msCM[ii],color=colsCM[ii],
#                 label=CMnamesAll[ii],linestyle='none')

#ax.xaxis.tick_top()
ax.set_xticks(xTick)
ax.set_xticklabels(CMnamesAllShort)

plt.savefig(figPath+'evi.pdf',bbox_inches='tight')
#plt.show()
plt.close()

## VERSION 2

#fig,axs = plt.subplots(1,1,figsize=(5.5,3.5)) # ,sharey=True)
#plt.subplots_adjust(right=0.7)
ax = axs

fig,axs = plt.subplots(1,2,figsize=(8.5,3.5),gridspec_kw={'width_ratios': [5, 4]}) # ,sharey=True)

ax = axs[0]

xTick = list(range(len(evidenceVals)))


ax.xaxis.tick_top()
ax.set_xticks(xTick)
ax.set_xticklabels(CMnamesAllShort)
#ax.set_ylabel('$\\log Z$')

for ii,ee in enumerate(postLikAll) :
    if ii == 0 : lab='${\\cal L}_{\\rm post}$'
    else : lab=None
    ax.plot(ii,ee,marker='o',ms=8,color='C0',
                 label=lab,linestyle='none')
    #ax.errorbar(ii,ee,yerr=postLikStdAll[ii],marker='o',ms=8,color='C0',
    #              label=lab,linestyle='none')

for ii,ee in enumerate(evidenceVals) :
    if ii == 0 : lab='$\\log Z$'
    else : lab=None
    ax.plot(ii,ee,marker='s',ms=8,color='black',
                  label=lab,linestyle='none')

yMin = -295
yMax = -265

ax.set_xlim(-0.5,xTick[-1]+0.5)
ax.set_ylim(yMin,yMax)

#ax = axs[1]
#ax.set_ylabel('${\\cal L}_{\\rm post}$')

#ax.set_ylim(likMin,likMax)


#ax.xaxis.tick_top()
#ax.set_xticks(xTick)
#ax.set_xticklabels(CMnamesAllShort)

ax.legend(#loc='upper left',bbox_to_anchor=(1.02,1.0),
          handlelength=0.5,)


ax = axs[1]

for ii,ee in enumerate(postPriAll) :
    if ii == 0 : lab='${\\cal P}_{\\rm post}$'
    else : lab=None
    ax.plot(ii,ee,marker='X',ms=8,color='C2',
                label=lab,linestyle='none')
    #ax.errorbar(ii,ee,yerr=postPriStdAll[ii],marker='X',ms=8,color='C2',
    #             label=lab,linestyle='none')

xTick = list(range(len(postPriAll)))
ax.xaxis.tick_top()
ax.set_xticks(xTick)
ax.set_xticklabels(CMnamesAllShort)

yMin = 75
yMax = 82

ax.set_xlim(-0.5,xTick[-1]+0.5)
ax.set_ylim(yMin,yMax)

ax.legend(loc='lower right',
          handlelength=0.5,)

plt.savefig(figPath+'evi2.pdf',bbox_inches='tight')
#plt.show()
plt.close()
