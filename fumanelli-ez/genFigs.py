#!/usr/bin/env python
# coding: utf-8

## python script for making Figs

## output  in ../finalFigs
### plotCM.pdf
### at_sketch.pdf
### mobility.pdf
### cfr_a.pdf

import numpy as np
from   matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm
import time

import pyross
import pickle
import pprint

import scipy.stats

from ew_fns import *
import expt_params_local
import model_local

## we quote figsize numbers as if they were cm (in fact they are inches)
## this means 20 is ok as default font
plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = 'serif'
plt.rc('text', usetex=True)


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

numCohorts = 16

## FIG : contact matrices

[fi,N,Ni,resident_data_2018_raw] = readPopData('../../data/EWAgeDistributedNew.csv',numCohPopData=19,numCohorts=16)
#plt.plot(fi) ; plt.show() ; plt.close()

matList = []
for chooseCM in ['fumanelliEtAl','premEtAl','flat']:
    contactBasis = setupCM(chooseCM,numCohorts,fi,verbose=False,pltAuto=False)
    myCM = contactBasis.constant_contactMatrix()
    matList.append(myCM)

import matplotlib as mpl

cmap = mpl.cm.BuPu
#norm = mpl.colors.Normalize(vmin=0, vmax=6)
norm = mpl.colors.LogNorm(vmin=0.01, vmax=10)

fig,axs = plt.subplots(1,4,figsize=(8.5,3.5),
                       gridspec_kw={'width_ratios': [1,1,1,0.1]})
plt.subplots_adjust(left=0.06,right=0.9,wspace=0.3,bottom=0.03)

## cohort age ranges
cohRanges = [ [x,x+4] for x in range(0,75,5) ]
cohLabs = ["{l:d}-{u:d}".format(l=low,u=up) for [low,up] in cohRanges ]
cohLabs.append("75+")

tickAges = [10,30,50,70]
tickPos = [-0.5 + xx/5 for xx in tickAges]

for ii,CM in enumerate(matList) :
    ax = axs[ii]
    ax.matshow(CM(0),cmap=cmap,norm=norm)

    ax.set_xlabel('age')
    ax.xaxis.set_label_position('top')
    ax.set_xticks(tickPos)
    ax.set_xticklabels(tickAges)
    ax.set_yticks(tickPos)
    ax.set_yticklabels(tickAges)

cb = mpl.colorbar.ColorbarBase(axs[-1],cmap=cmap,norm=norm)
plt.savefig(figPath+'plotCM.pdf')
plt.close()

## FIG sketches of a(t) for interventions

fig,axs = plt.subplots(1,2,figsize=(8.5,3.5)) ## imagine figsize is cm and use font size 16...
plt.subplots_adjust(left=0.1,right=0.78,bottom=0.19,top=0.95)

ax = axs[0]

finTime = 10

pltArgs = {'lw':3}

loc = 2.5
width = 1.0
aF = 0.25
xVals = np.linspace(0,finTime,100)
aVals = [ np.interp(xx,[loc-width,loc+width],[1,aF]) for xx in xVals ]
ax.plot(xVals,aVals,**pltArgs)
ax.plot(loc,np.interp(loc,[loc-width,loc+width],[1,aF]),'o',color='C0')
ax.set_xlim(0,finTime)
ax.set_ylim(0,1.1)
ax.set_xlabel('$t$')
ax.set_ylabel('$a_i(t)$')
ax.set_yticks([0,aF,1])
ax.set_yticklabels(['0',r'$a^{\rm F}_i$','1'])

ax = axs[1]
ease= 0.3
xVals = np.linspace(0,finTime,100)
aVals = [ np.interp(xx,[loc-width,loc+width,finTime],[1,aF,1-(1-aF)*(1-ease)]) for xx in xVals ]
ax.plot(xVals,aVals,**pltArgs)
ax.plot(loc,np.interp(loc,[loc-width,loc+width],[1,aF]),'o',color='C0')
ax.set_xlim(0,finTime)
ax.set_ylim(0,1.1)
#ax.set_ylabel('$a_i(t)$')
ax.set_xlabel('$t$')
ax.set_yticks([aF,1-(1-aF)*(1-ease),1])
ax.set_yticklabels([r'$a^{\rm F}_i$',r'$a^{\rm F}_i+r(1-a^{\rm F}_i)$','1'])

ax.yaxis.set_tick_params(labelright=True,right=True,labelleft=False,left=False)
plt.savefig(figPath+'at_sketch.pdf')
plt.close()

## google mobility plot

dataPath = '../../data/'
GMData=pd.read_csv(dataPath+'uk_mobility_data_national_level.csv')

## we are going to compute rolling 7-day averages of activity
startDay = 20
startDate = GMData.iloc[startDay]['date']
print('GMob average starting on date',startDate)

numWeeks = 12
daysPerWeek = 7
numDays = numWeeks*daysPerWeek

## check that data set us long enough for requested numWeeks
numDays = np.minimum( numDays, len(GMData)-4-startDay )
print('computing',numDays,'days')

## do the average

## each element of homeDat (etc) will be [t,x],
##   where t is time index and x is 7-day rolling average
homeDat = []
workDat = []
transitDat = []
retailDat = []
grocDat = []

## list of datasets
allDat = [homeDat,
          workDat,
          transitDat,
          retailDat,
          grocDat,
         ]

labels = ['residential_percent_change_from_baseline',
          'workplaces_percent_change_from_baseline',
          'transit_stations_percent_change_from_baseline',
          'retail_and_recreation_percent_change_from_baseline',
          'grocery_and_pharmacy_percent_change_from_baseline'
         ]
shortLabels = ['home','work','transit stations','retail/recreation','grocery/pharmacy']

## here is the rolling average
for ii in range(startDay,startDay+numDays):
    for jj,dat in enumerate(allDat) :
        dat.append( [ ii-startDay,
                      np.mean(GMData.iloc[ii-3:ii+4][labels[jj]] ) ] )

## rescale data as relative activity
for ss in allDat :
    for dd in ss :
        dd[1]=(100+dd[1])/100

## make a plot
fig,ax = plt.subplots(1,1,figsize=(8,5))
plt.subplots_adjust(left=0.13,right=0.6,bottom=0.15)
for ii,ss in enumerate(allDat) :
    ax.plot([dd[0]/7.0 for dd in ss],  ## time in weeks
             [dd[1] for dd in ss],      ## value
             'o-',label=shortLabels[ii],ms=3)
ax.set_xlabel('time (weeks)')
ax.set_ylabel('relative activity')
ax.set_ylim(0,1.4)
ax.legend(bbox_to_anchor=(1, 1.0),handlelength=1.0)
plt.savefig(figPath+'mobility.pdf')
#plt.show() ;
plt.close()


## FIG alpha and CFR


fig,axs = plt.subplots(1,2,figsize=(8.5,3.5)) ## imagine figsize is cm and use font size 16...
plt.subplots_adjust(left=0.12,right=0.95,bottom=0.19,top=0.95,wspace=0.3)

ax = axs[0]

ax.set_yscale('log')
ax.plot(modParams['cfr'],'o-')

yesTick = [0,4,8,12,15]
ax.set_xticks(yesTick)
ax.set_xticklabels([cohLabs[xx] for xx in yesTick])
ax.set_ylabel('$f_i$')
ax.set_xlabel('age')
ax.set_ylim(1e-4,0.1)

ax = axs[1]

ax.plot(modParams['alpha'],'o-')

yesTick = [0,4,8,12,15]
ax.set_xticks(yesTick)
ax.set_xticklabels([cohLabs[xx] for xx in yesTick])
ax.set_ylabel('$\\alpha_i$')
ax.set_xlabel('age')
ax.set_ylim(0,0.65)


plt.savefig(figPath+'cfr_a.pdf')
#plt.show() ;
plt.close()
