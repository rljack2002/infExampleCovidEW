#file -- expt_params_local.py

## parameters that (might) vary between model variants

def getLocalParams() :

    dataPath = "../data/"

    exptParams = {
        'inferBetaNotAi' : True,               ## use True here, False is for legacy
        'exCare' :         True,               ## exclude deaths in care homes
        'chooseCM' : "fumanelliEtAl",          ## contact matrix

        'pikFileRoot' : "ewMod" ,   ## for storing results

        'dataFile' : dataPath+"OnsData.csv" ,             ## deaths
        'careFile' : dataPath+"CareHomes.csv" ,           ## care home deaths
        'popFile'  : dataPath+"EWAgeDistributedNew.csv",  ## where do we read population?
        'numCohortsPopData' : 19,                         ## cohorts in popFile

        'numCohorts' : 16,  ## age cohorts

        'timeZero' : 0,  ## inference window: array index of first data point (zero is 6-mar)
        'timeLast' : 8,  ## inference window: timeLast - timeZero is num of data points
                         ## so last array index is timeZero+timeLast-1
    
        'forecastTime' : 3,   ## for post-inference run and comparison
    
        ## which classes have "special" priors (eldest cohort only)
        'freeInitPriors' : [ 'E','A','Is1','Is2','Is3' ],
    
        'estimatorTol' : 1e-8,  ## tolerance param (rtol_det in estimator setup)
    
        'infOptions' : {    ## these are for latent_infer
                        'global_max_iter' : 1500,  ## should be enough to converge
                        'global_atol' :      1.0,  ## seems reasonable
                        'local_max_iter' :   400,  ## value here not too crucial
                        'ftol' :            5e-5,  ## this is 1e-2 on a value of 2e2 == 200
                        'cma_processes' :   None,  ## use default
                        'cma_population' :    32,  ## seems ok for 40 params or so
        },
    }
    return exptParams


