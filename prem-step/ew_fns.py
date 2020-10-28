#file -- ew_fns.py

## generic functions for eng-wal model

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import pprint

import pyross

## population data
def readPopData(file,numCohPopData=1,numCohorts=1) :
    resident_data_2018_raw = pd.read_csv(file)

    ## number in each cohort
    Ni=np.zeros(numCohorts)
    for i in range(numCohorts):
        Ni[i]=resident_data_2018_raw['Num'][i]
    
    ## combine older groups into last cohort
    for i in range(numCohorts, numCohPopData):
        Ni[-1] += resident_data_2018_raw['Num'][i]

    ## total
    N = np.sum(Ni)
    ## fractions
    fi = Ni/N
    
    return [fi,N,Ni,resident_data_2018_raw]

## model-specification
def getModThreeStage() :
    model_spec = {
      "classes" : ["S", "E", "A", "Is1", "Is2", "Is3", "Im"],

      "S" : {  ## susceptible
          "linear"    : [], # [["Ia", "immune*gammaIa"]],  ## recovery without immunity
          "infection" : [ ["A", "-beta"],
                          ["Is1", "-beta"],
                          ["Is2", "-betaLate"],
                          ["Is3", "-betaLate"] ]
      },

      "E" : {  ## exposed (non-infectious)
          "linear"    : [ ["E", "-gammaE"] ],
          "infection" : [ ["A", "beta"],
                          ["Is1", "beta"],
                          ["Is2", "betaLate"],
                          ["Is3", "betaLate"] ]
      },

      "A" : {  ## activated (first infectious stage)
          "linear"    : [ ["E", "gammaE"],
                          ["A", "-gammaA"] ],
          "infection" : [ ]
      },

      "Is1" : { ## infectious (can be mild or severe)
          "linear"    : [ ["A", "gammaA"],
                          ["Is1", "-alphabar*gammaIs1"],  ## transfer to late-stage infectious
                          ["Is1", "-alpha*gammaIs1"] ],   ## recovery from mild symptoms
          "infection" : [ ]
      },

      "Is2" : { ## late-stage infectious with more-severe symptoms stage 1
          "linear"    : [ ["Is1", "alphabar*gammaIs1"],
                          ["Is2", "-gammaIs2"] ],
          "infection" : [ ]
      },

      "Is3" : { ## late-stage infectious with more-severe symptoms stage 2
          "linear"    : [ ["Is2", "gammaIs2"],
                          ["Is3", "-cfrbar*gammaIs3"],
                          ["Is3", "-cfr*gammaIs3"] ],
          "infection" : [ ]
      },

      "Im" : { ## deceased
          "linear"    : [ ["Is3", "cfr*gammaIs3"] ],
          "infection" : [ ]
      }
  }
    return model_spec

## these are epidemiological params
## they include priors and some fixed params (eg CFR)
def getParamsThreeStage(numCohorts,popDataAll,daysPerWeek,verbose=True) :
    latentTime  = 3.0 ## days, Milo says 3-4
    preSympTime = 2.5 ## days, Milo says 5-6 before symptoms (latent+pre)
    infSympTime = 3.0 ## days, Milo says 4-5 infectious period (pre+inf)
    postSympTime= 20 - (preSympTime+infSympTime)  ## Jakub had 17.8 + 2.7
    postOne = postSympTime/2.0
    postTwo = postSympTime/2.0

    if verbose : print('**times',latentTime,preSympTime,infSympTime,postSympTime,postOne,postTwo)

    gammaE   = daysPerWeek/latentTime
    gammaA   = daysPerWeek/preSympTime
    gammaIs1 = daysPerWeek/infSympTime
    gammaIs2 = daysPerWeek/postOne
    gammaIs3 = daysPerWeek/postTwo

    ## prior for "prob to be infected on contact" 
    ## the 0.2 is very arbitrary here
    beta = 0.2 * np.array([0.2, 0.2, 0.2, 0.3,
                           0.4, 0.5, 0.6, 0.7,
                           0.8, 0.9, 1,   1,
                           1,   1,   1,   1] )

    ## this deals with the fact that contact matrices are "per day"
    ## but the time unit is a week (so beta here is eta*beta in notation of paper)
    beta *= daysPerWeek

    ## how much less infectious is a 'late-stage' individual
    ## (could also consider zero here)
    betaLateFactor = 0.1 

    ## these numbers are hardcoded, would need other kinds of data to infer them

    ## num cohorts in these hardcoded data?
    ## (must be aligned with popDataAll)
    numCohortsTabData = 19
   
    ## these are copy-pasted from Italian data.ipynb 
    ## they are aligned with our cohorts 

    # prob for "very mild" or "asymptomatic"
    alphaM16 = [ 0.41252774, 0.51985922, 0.56484453, 0.5171226 , 0.45909986,  ## 0-24 in this row
                 0.41789711, 0.39351434, 0.3609665 , 0.32025358, 0.28422119,  ## 25-49
                 0.25286934, 0.21460018, 0.16941369, 0.13699457, 0.11734282,  ## 50-74
                 0.12898587 ]

    # prob for death, given that symptoms were "clinical" (ie more than very mild)
    cfrM16 = [ 0.00017314, 0.00025375, 0.00037189, 0.00054504, 0.00079881, 
               0.00117073, 0.00171581, 0.00251468, 0.00368548, 0.00540141, 
               0.00791625, 0.01160198, 0.01700374, 0.02492051, 0.03652325, 
               0.07315813 ]
           
    alpha = np.array( alphaM16 )
    cfr   = np.array( cfrM16 )

    ## deleted some old code here that was used to realign CFR etc with our cohorts

    ## plot for illustration (if req'd)
    if verbose :
        plt.plot(alpha,'o-',label='alpha (= mild)')
        plt.legend() ; plt.show() ; plt.close()

        plt.yscale('log')
        plt.plot(cfr,'o-',label='cfr')
        plt.plot(cfr*(1-alpha),'o-',label='ifr')
        plt.legend() ; plt.show() ; plt.close()

        print('** cfr\n',cfr,'\n')
        print('** alpha\n',alpha,'\n')

    modParams = {
        "beta" :    beta,
        "betaLateFactor": betaLateFactor,
        "gammaE":   gammaE,
        "gammaA":   gammaA,
        "gammaIs1": gammaIs1,
        "gammaIs2": gammaIs2,
        "gammaIs3": gammaIs3,
        "alpha":alpha,
        "cfr":cfr,
    }

    return modParams

## this function sets the rate parameters of the model from the epidemiological parmaeters
def paramMap(modParams) :
    rateParams = {
        "beta"    : modParams["beta"],
        "betaLate": modParams["beta"] * modParams["betaLateFactor"],
        "gammaE"  : modParams["gammaE"],
        "gammaA"  : modParams["gammaA"],
        "alpha*gammaIs1" : modParams["alpha"] * modParams["gammaIs1"],
        "alphabar*gammaIs1" : (1-modParams["alpha"]) * modParams["gammaIs1"],
        "gammaIs2" : modParams["gammaIs2"],
        "cfr*gammaIs3" : modParams["cfr"] * modParams["gammaIs3"],
        "cfrbar*gammaIs3" : (1-modParams["cfr"]) * modParams["gammaIs3"],
    }
    return rateParams

## setup contact matrix
def setupCM(chooseCM,numCohorts,fi,verbose=True,pltAuto=True) :

    if chooseCM != 'flat' :
        CH0, CW0, CS0, CO0 = pyross.contactMatrix.UK(source=chooseCM)
        sizeCM = np.shape(CH0)[0]
        #print(sizeCM)
    else :
        sizeCM = numCohorts

    fumRescale = 3.0

    zeroC = np.zeros( (numCohorts,numCohorts) )
    allC =  np.zeros( (numCohorts,numCohorts) )
    if chooseCM == 'fumanelliEtAl' :
        rescaleFactor = fumRescale  ## this is arbitrary because fumanelli CM is not normalised

        ## setup 5-year cohorts
        cohortGap = 5
        ageRanges = [ range(c*cohortGap,(c+1)*cohortGap) for c in range(0,numCohorts-1) ]
        ageRanges.append( range((numCohorts-1)*cohortGap,sizeCM) )
        allCRaw = CO0+CS0+CH0+CW0
        for i in range(numCohorts) :
            for j in range(numCohorts) :
                allC[i,j] = np.sum( allCRaw[ageRanges[i],:][:,ageRanges[j]] )

        tempC = np.copy( allC.T )
        tempC /= (fi * sizeCM * numCohorts * rescaleFactor)
        allC = tempC.T

    elif chooseCM == 'premEtAl' :
        allC = CO0+CS0+CH0+CW0
        print('typC',np.mean(allC))

    ## flat means proportinal mixing
    elif chooseCM == 'flat' :
        cH0, cW0, cS0, cO0 = pyross.contactMatrix.UK(source='premEtAl')
        cO0 += cH0 + cW0 + cS0
        typCprem = np.mean(cO0)
        allC = typCprem * fi*numCohorts * np.ones((numCohorts,numCohorts))
        typCflat = np.mean(allC)
        if verbose : print('typC: flat',typCflat,'prem',typCprem)
    else:
        assert False, 'only fumanelli / prem / prop-mixing CMs supported so far'

    ## this object is what is sometimes called 'generator' in pyRoss
    contactBasis = pyross.contactMatrix.ContactMatrixFunction(zeroC,zeroC,zeroC,allC)

    if verbose :
        plt.rcParams.update({'figure.autolayout': False})
        print("log(CM), this is not symmetric")
        plt.matshow(np.log(allC))
        plt.colorbar()
        plt.show()
        plt.close()
        plt.rcParams.update({'figure.autolayout': pltAuto})

    return contactBasis


## read in the death data
def readData(file,numCohorts,exCare,careFile=None,maxWeeks=None,verbose=True) :
    deathsFromFile = pd.read_csv(file, sep=',')

    deathsFromFile = deathsFromFile.T

    fileShape = deathsFromFile.shape
    cohortsFromFile = fileShape[1]-1

    firstWeek = 11  ## week ending 6-mar
    lastWeek = fileShape[0]-2
    if maxWeeks != None and lastWeek > firstWeek + maxWeeks -1 :
        lastWeek = firstWeek + maxWeeks -1
    if verbose :
      print('** first week',firstWeek,deathsFromFile[:].values[firstWeek])
      print('** last week ',lastWeek, deathsFromFile[:].values[lastWeek])

    numWeeks = lastWeek - firstWeek + 1   ## fenceposts
    deathWeeklyData = np.zeros((numCohorts,numWeeks))
    for i in range(numCohorts):
        deathWeeklyData[i] = deathsFromFile[i+1].values[firstWeek:lastWeek+1].astype(int)+0.01
    
    ## combine elderly into last cohort
    for i in range(numCohorts,cohortsFromFile):
        deathWeeklyData[-1] += deathsFromFile[i+1].values[firstWeek:lastWeek+1].astype(int)+0.01
    if verbose :
      print('** weekly deaths youngest cohort\n',deathWeeklyData[0])
      print('** weekly deaths oldest cohort\n',deathWeeklyData[-1])

    ## subtract care-home deaths
    if exCare:
        if verbose : print('** subtracting care home deaths...')
        careDeathsFromFile = pd.read_csv(careFile, sep=',') # , header=None)
        careDeathsFromFile = careDeathsFromFile.T
        ## we need an offset to line up the data
        if deathsFromFile[0].values[firstWeek] != careDeathsFromFile[0].values[firstWeek-1] :
            print(deathsFromFile[0].values[firstWeek],careDeathsFromFile[0].values[firstWeek-1])
            assert False, 'bad dates in care data, do not align properly ??'
        deathWeeklyData[-1] -= careDeathsFromFile[1].values[firstWeek-1:lastWeek].astype(float)
        if verbose :
          print('** weekly care deaths\n',careDeathsFromFile[1].values[firstWeek-1:lastWeek].astype(float) )
          print('** adjusted weekly deaths oldest cohort\n',deathWeeklyData[-1])

    deathCumulativeData = np.zeros((numCohorts,numWeeks))
    for ww in range(1,1+numWeeks) :
        deathCumulativeData[:,ww-1] = np.sum(deathWeeklyData[:,:ww],axis=1)
    if verbose : ('** cumul deaths oldest cohort\n',deathCumulativeData[-1])

    return deathCumulativeData

## helper function to make filter matrix
def fltrClasses(toFltr,M,model_spec,verbose=True) :
    classList = []
    for cc in model_spec['classes'] :
        if cc in toFltr :
            classList.append(1)
        else:
            classList.append(0)
    if verbose : print('** create fltr, list : ',classList)
    return np.kron((classList), np.identity(M))

## piecewise-linear step function (for intervention)
def stepLikeFn(t, width, loc):  ## from -1 to 1
    a = loc - (width/2)
    b = loc + (width/2)
    return np.interp(t,[a,b],[-1,1])

## these functions are used for NPIs

## intervention with flexible initial and final vals
def interventionFnInitFinal(t, M, width=1, loc=0, aO_f=0, aO_i=0):
    aW = 0.0 # (1-stepLikeFn(t, width, loc))/2*(1-aW_f) + aW_f
    aS = 0.0 # (1-stepLikeFn(t, width, loc))/2*(1-aS_f) + aS_f
    aO = (1-stepLikeFn(t, width, loc))/2*(aO_i-aO_f) + aO_f
    aW_full = np.full((2, M), aW) # must return the full (2, M) array
    aS_full = np.full((2, M), aS)
    aO_full = np.full((2, M), aO)
    return aW_full, aS_full, aO_full

## intervention with initial val == 1
def interventionFnFinalOnly(t, M, width=1, loc=0, aO_f=0):
    aW = 0.0 # (1-stepLikeFn(t, width, loc))/2*(1-aW_f) + aW_f
    aS = 0.0 # (1-stepLikeFn(t, width, loc))/2*(1-aS_f) + aS_f
    aO = (1-stepLikeFn(t, width, loc))/2*(1-aO_f) + aO_f
    aW_full = np.full((2, M), aW) # must return the full (2, M) array
    aS_full = np.full((2, M), aS)
    aO_full = np.full((2, M), aO)
    return aW_full, aS_full, aO_full

## intervention with initial val == 1 that increases linearly after end of step
## our idea is that aO_f is vector but easeFrac is scalar
##   (maybe easeFrac can be vector too but this is not tested)
def interventionFnEase(t, M, width=1, loc=0, aO_f=0, easeFrac=0) : # , endEase=0):
    aW = 0.0
    aS = 0.0

    endEase = 10.0  ## hardcoded(!) , in weeks

    startStep = loc - 0.5*width
    endStep = loc + 0.5*width
    if t > endStep :
        ## interp from aO_f at endStep so that we have relaxed by easeFrac at endEase
        scaleFact = 1 - ( easeFrac * (t-endStep)/(endEase-endStep) )
        aO = 1- (1-aO_f)*scaleFact
    else :
        ## this is the step
        aO = (1-aO_f)*np.interp(t,[startStep,endStep],[1,0]) + aO_f
        
    aW_full = np.full((2, M), aW) # must return the full (2, M) array
    aS_full = np.full((2, M), aS)
    aO_full = np.full((2, M), aO)
    
    return aW_full, aS_full, aO_full

### intervention with initial val == 1 that increases non-linearly after end of step
### (not used)
#def interventionFnLogEase(t, M, width=1, loc=0, aO_f=0, expEaseFrac=0) : # , endEase=0):
#    aW = 0.0
#    aS = 0.0
#
#    endEase = 10.0  ## hardcoded(!) , in weeks
#
#    startStep = loc - 0.5*width
#    endStep = loc + 0.5*width
#    if t > endStep :
#        ## interp from aO_f at endStep so that we have relaxed by easeFrac at endEase
#        scaleFact = 1 - ( np.log(expEaseFrac) * (t-endStep)/(endEase-endStep) )
#        aO = 1- (1-aO_f)*scaleFact
#    else :
#        ## this is the step
#        aO = (1-aO_f)*np.interp(t,[startStep,endStep],[1,0]) + aO_f
#
#    aW_full = np.full((2, M), aW) # must return the full (2, M) array
#    aS_full = np.full((2, M), aS)
#    aO_full = np.full((2, M), aO)
#
#    return aW_full, aS_full, aO_full


## priors for epidemiological params
def getPriorsEpiThreeStage(modParams,inferBetaNotAi=True) :
    priorsAll = { }

    ## this is true
    if inferBetaNotAi :
        beta = modParams['beta']
        priorsAll['beta'] = {
                    'mean': beta,
                    'std' : beta*0.5,
                    'bounds' : [[bb*0.1,bb*10.0] for bb in beta]
                }

    ## infectiousness of Is2
    betaLateFactor = modParams['betaLateFactor']
    priorsAll['betaLateFactor'] = {
                'mean': betaLateFactor,
                'std' : betaLateFactor/4.0,
                'bounds' : [betaLateFactor/10.0,1.0]
            }

    ## rates
    def tightPrior(mean) :
        res = {
            'mean':mean,
            'std': 0.1*mean,
            #'bounds': [0.7*mean, 1.4*mean],  ## for lognorm
            'bounds': [0.6*mean, 1.4*mean],   ## for truncnorm
            'prior_fun' : 'truncnorm'
        }
        return res

    priorsAll['gammaE'] = tightPrior(modParams['gammaE'])
    priorsAll['gammaA'] = tightPrior(modParams['gammaA'])
    priorsAll['gammaIs1'] = tightPrior(modParams['gammaIs1'])
    priorsAll['gammaIs2'] = tightPrior(modParams['gammaIs2'])
    priorsAll['gammaIs3'] = tightPrior(modParams['gammaIs3'])

    return priorsAll

## params for NPIs which are common to all variants
def priorsConCommon(daysPerWeek) :

    aO_g = 1.0
    aO_s = 0.5
    aO_b = [1e-2,1e1]

    lockFactor_g = 5.0

    locTime_g = 17.0/daysPerWeek

    return [aO_g,aO_s,aO_b,lockFactor_g,locTime_g]

## prior params for step-like-NPI
def getPriorsControl(numCohorts,daysPerWeek,
                     inferBetaNotAi=True,
                     widthPriorMean=4.0,widthPriorStd=1.0,  ## in days
                     ) :
    [aO_g,aO_s,aO_b,lockFactor_g,locTime_g] = priorsConCommon(daysPerWeek)
    
    locTime_s = 1.0/daysPerWeek
    locTime_b = [ 1.0/daysPerWeek , 30.0/daysPerWeek ]

    width_g = widthPriorMean/daysPerWeek      ## note this is "full-width" not half-width
    width_s = widthPriorStd /daysPerWeek      ## with mean==4 and std==1, had MCMC problems with v broad widths
    width_b = [ 0.1/daysPerWeek , 20.0/daysPerWeek ]

    priorsCon = {
        'aO_f':{
            'mean': [aO_g/lockFactor_g]*numCohorts,
            'std' : [aO_s/lockFactor_g]*numCohorts,
            'bounds': [ [ x/lockFactor_g for x in aO_b ] ]*numCohorts
        },
        'loc':{
            'mean': locTime_g,
            'std' : locTime_s,
            'bounds': locTime_b,
            'prior_fun' : 'truncnorm'
        },
        'width':{
            'mean': width_g,
            'std' : width_s,
            'bounds': width_b,
            'prior_fun' : 'truncnorm'
        }
    }
    #print(priorsCon_vector)

    if inferBetaNotAi :
        interventionFn = interventionFnFinalOnly
    else :
        interventionFn = interventionFnInitFinal
        priorsCon['aO_i'] = {
            'mean': [aO_g]*numCohorts,
            'std' : [aO_s]*numCohorts,
            'bounds': [aO_b]*numCohorts
        }

    return [priorsCon,interventionFn]

## NPI-with- nonlinear -easing (not used)
#def getPriorsControlGMobLog(numCohorts,daysPerWeek,
#                     widthPriorMean=4.0,widthPriorStd=1.0,  ## in days
#                     ) :
#
#  [aO_g,aO_s,aO_b,lockFactor_g,locTime_g] = priorsConCommon(daysPerWeek)
#
#
#  locTime_s = 1.0/daysPerWeek
#  locTime_b = [ 1.0/daysPerWeek , 30.0/daysPerWeek ]
#
#  ## this object is logNormal dist, its log is the fraction we want, which is norm distributed
#  expEaseFrac_g = np.exp(0.2)
#  expEaseFrac_s = 0.1  ## we want the exp to be ~ between 1.1 and 1.3 which means easeFrac between 0.1 and 0.3
#  expEaseFrac_b = [ np.exp(0.05) , np.exp(0.5) ]
#
#  width_g = widthPriorMean/daysPerWeek      ## note this is "full-width" not half-width
#  width_s = widthPriorStd /daysPerWeek      ## with mean==4 and std==1, had MCMC problems with v broad widths
#  width_b = [ 0.1/daysPerWeek , 20.0/daysPerWeek ]
#
#  priorsCon = {
#      'aO_f':{
#          'mean': [aO_g/lockFactor_g]*numCohorts,
#          'std' : [aO_s/lockFactor_g]*numCohorts,
#          'bounds': [ [ x/lockFactor_g for x in aO_b ] ]*numCohorts
#      },
#      'loc':{
#            'mean': locTime_g,
#            'std' : locTime_s,
#            'bounds': locTime_b,
#      },
#      'expEaseFrac':{
#            'mean': expEaseFrac_g,
#            'std' : expEaseFrac_s,
#            'bounds': expEaseFrac_b,
#      },
#      'width':{
#            'mean': width_g,
#            'std' : width_s,
#            'bounds': width_b,
#      },
#  }
#
#  interventionFn = interventionFnLogEase
#  return [priorsCon,interventionFn]

## prior params for NPI-with-easing
def getPriorsControlGMob(numCohorts,daysPerWeek,
                     widthPriorMean=4.0,widthPriorStd=1.0,  ## in days
                     ) :

  [aO_g,aO_s,aO_b,lockFactor_g,locTime_g] = priorsConCommon(daysPerWeek)


  locTime_s = 1.0/daysPerWeek
  locTime_b = [ 1.0/daysPerWeek , 30.0/daysPerWeek ]

  easeFrac_g = 0.2
  easeFrac_s =   easeFrac_g/2.0
  easeFrac_b = [ 0.05 , 0.5 ]

  width_g = widthPriorMean/daysPerWeek      ## note this is "full-width" not half-width
  width_s = widthPriorStd /daysPerWeek      ## with mean==4 and std==1, had MCMC problems with v broad widths
  width_b = [ 0.1/daysPerWeek , 20.0/daysPerWeek ]

  priorsCon = {
      'aO_f':{
          'mean': [aO_g/lockFactor_g]*numCohorts,
          'std' : [aO_s/lockFactor_g]*numCohorts,
          'bounds': [ [ x/lockFactor_g for x in aO_b ] ]*numCohorts
      },
      'loc':{
            'mean': locTime_g,
            'std' : locTime_s,
            'bounds': locTime_b,
            'prior_fun' : 'truncnorm'
      },
      'easeFrac':{
            'mean': easeFrac_g,
            'std' : easeFrac_s,
            'bounds': easeFrac_b,
            'prior_fun' : 'truncnorm'
      },
      'width':{
            'mean': width_g,
            'std' : width_s,
            'bounds': width_b,
            'prior_fun' : 'truncnorm'
      },
  }

  interventionFn = interventionFnEase
  return [priorsCon,interventionFn]

## priors for initial conditions
def getInitPriors(model_spec,numCohorts,N,freeInitPriors=[],fixR=False) :

    coeff_guess = 5e-4      ## prior mean for coefficient of leading mode
    coeff_std_shape = 0.5   ## std dev in units of mean, was 1.0, previously was 2.0

    ## here we deal with the "special" priors (not determined by leading mode)
    ## create empty dictionary
    initFreeGuess = {}

    ## populate dictionary (rather arbitrary priors)
    initFreeGuess['E']  = 2000.0
    initFreeGuess['A']  = 1200.0
    initFreeGuess['Is1'] = 300.0
    initFreeGuess['Is2']= 100.0 * 0.6
    initFreeGuess['Is3']= 100.0 * 0.4

    initFreeUncertainty  = 0.5  ## std / mean for these compartments

    if fixR : ## this is the final method, a bit complicated
        
        nClass  = len(model_spec['classes'])

        ## there is a filter matrix to determine initial vals from linear mode
        matFltr = np.zeros((nClass-1,nClass))

        for ii in range(nClass-1) : matFltr[ii,ii] = 1.0  ## diagonal for most classes
        for ii in range(nClass-1) : matFltr[0,ii] = 1.0   ## this means that we get R (not S) from linear mode

        ## kronecker product is because compartments are indexed by both class and cohort
        matFltr = np.kron(matFltr,np.eye(numCohorts))

        ## now we need another filter matrix for the "special" compartments (wich don't come from indpt mode)
        matFltrInd = []

        ## copy the matFltr
        matFltrMixed = np.copy(matFltr)
        offset = 0

        ## we have to move some rows of matFltrMixed into matFltrInd, to make these compartments "special"
        ## the freeInitPriors must be in class order else this loop fails
        for cc in freeInitPriors  :

            ## which compartment do we consider?
            indSpecial = model_spec['classes'].index(cc)
            elem = (numCohorts*indSpecial) + (numCohorts-1)  ## index of compartment

            ## add a row to the independentfltr matrix
            matFltrInd.append(matFltrMixed[elem-offset])

            ## remove the corresponding row for the linMode matrix
            matFltrMixed = np.delete(matFltrMixed,elem-offset,axis=0)

            ## this offset is because we have removed a row from the matFltrMixed so its indices have changed
            offset += 1 

        ## convert matFltrInd list to numpy matrix
        matFltrInd = np.array(matFltrInd)

        ## now the filters are finally done, deal with prior numbers

        ## mean values for independent compartments
        specMean = [ initFreeGuess[key]/N for key in freeInitPriors ] 

        ## this is the dictionary required by inference
        initPriors = {
            'lin_mode_coeff' : {
                'mean' : coeff_guess,
                'std' : coeff_std_shape * coeff_guess,
                'bounds' : [coeff_guess/100.0,coeff_guess*100.0],
                'fltr': matFltrMixed
            } ,
            'independent' : {
                'mean' : [m for m in specMean],
                'std' : [m*initFreeUncertainty for m in specMean],
                'bounds' : [[m/10.0,m*10.0] for m in specMean],
                'fltr': matFltrInd
            }

        }
        
    else :
        assert False, 'getInitPriors: fixR should be True'

    return initPriors
