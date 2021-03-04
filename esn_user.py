#Misc. 
import threading
import sys
import time
#Backends
import numpy as np
import torch
#Save data structure
import h5py
#ESN
import easyesn
from easyesn import PredictionESN, BaseESN
from easyesn import backend as B

#functions & class
path_to_files = '/home/flhe/nextcloud/Promotion/python/ESN/github/'
sys.path.insert(1, path_to_files)
from esn_fun import ImportData, NormalizeData, PreparePredictorData, PredictModel, BuildAndFitModel, ConvertNumpyToTorch, ComputeMSE, RunESNWithParams, SaveESNParams, SaveModelStudy
from esn_params import ESNParams


#--------------------------------------------------------------------------------------------------------
class ESNThread(threading.Thread):

    def __init__(self, threadID, esn_params, filepath_esn, study_parameters, nstudies, SetParameters, usingTorch):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.esn_params = esn_params
        self.filepath_esn = filepath_esn
        self.study_parameters = study_parameters
        self.nstudies = nstudies
        self.SetParameters = SetParameters
        self.usingTorch = usingTorch

    def run(self):
        print('Thread {0} starts.'.format(self.threadID))

        # easyesn v.0.1.6 has outdated seeding for torch backend --> set seed manually and set randomSeed to None
        if self.usingTorch:
            torch.manual_seed(self.esn_params.SeedID)             
            self.esn_params.randomSeed = None                        

        studytime_start = time.time()
        studytime_end = RunESNStudy(self.threadID,
                                    self.esn_params,
                                    self.filepath_esn, 
                                    self.study_parameters, 
                                    self.nstudies, 
                                    self.SetParameters)
        studytime = studytime_end -studytime_start
        print('Thread {0} done. Elapsed time {1}'.format(self.threadID, studytime))
#--------------------------------------------------------------------------------------------------------
def Recursion(iparam,iterators,study_parameters):
    ''' Iterates the iterators which are used to change the hyperparmeters. 
        Makes sure that all combinations of the parameters in study_parameters are used.

        INPUT:
            iparam           - index which defines the hyperparameter that is changed
            iterators        - iterator that defines the no. ESNs that have been run with changing one hyperparameter (study_parameters[iparam]) 
            study_parameters - tuple specifying the range of the parameters that are studies

    '''
    
    #if iterator associated with iparam hasn't reached final value --> increment
    if iterators[iparam] < len(study_parameters[iparam])-1:
        iterators[iparam] +=1
            
    #if iterator associated with iparam has reached final value --> reset iterator and increment/reset higher level iterator associated with iparam-1.
    else:
        iterators[iparam] = 0
        Recursion(iparam-1,iterators,study_parameters)
#--------------------------------------------------------------------------------------------------------

def RunESNStudy(threadID, esn_params, filepath_esn, study_parameters, nstudies, SetParameters):
    ''' USER-DEFINED ESN run. Can be used for studies.

        INPUT:
            threadID         - ID of the current thread executing this function. Usually corresponds to esn_params.SeedID
            esn_params       - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.
            filepath_esn     - path to which the hdf5 file of the ESN studies & model is saved to
            study_parameters - tuple specifying the range of the parameters that are studies
            nstudies         - total number of studies/ parameter settings
            SetParameters    - member functions that set the parameters specified in study_parameters in the ESNParams class object

        RETURN:
            time at which this function is done

        USED INDICES/ ITERATORS:
        itotal     - iterator that defines the total no. ESNs that have been run
        iparam     - index which defines the hyperparameter that is changed
        iterators  - iterator that defines the no. ESNs that have been run with changing one hyperparameter (study_parameters[iparam])
    '''
    
    #----------------------------------
    #Initialize quantities
    #----------------------------------
    study_dict = {}
    nstudyparameters = len(study_parameters)
    iterators = np.zeros([nstudyparameters], dtype=int)

    if len(SetParameters) != nstudyparameters:
        print('No. setter functions ({0}) does not match no. study parameters ({1}).\n Exiting...'.format(len(SetParameters), nstudyparameters))
        exit()


    #----------------------------------
    #Start Study
    #----------------------------------
    iparam = nstudyparameters-1
    for itotal in range(nstudies):
        if threadID == 0:
            #suppress output for other threads
            print('Thread {0}: Study {1}/{2}.'.format(threadID, itotal+1, nstudies))
            
        

        #Update iterators:      
        if itotal == 0:
            pass
        else:
            Recursion(nstudyparameters-1,iterators,study_parameters)
 
        #Update set of hyperparameters
        for iparam in range(nstudyparameters):
            ival = iterators[iparam]
            val = study_parameters[iparam][ival]
            
            SetParameters[iparam](val, study_dict)

        #Run ESN
        mse_train, mse_test, y_pred = RunESNWithParams(esn_params)
        SaveModelStudy(path_esn+filename_esn, esn_params, y_pred, mse_train, mse_test, itotal, study_dict)    

        
    return time.time()    
#--------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------#
#                                                                       #
#                               MAIN                                    #
#                                                                       #
#-----------------------------------------------------------------------#

if __name__ == '__main__':

    #Check easyesn backend 
    usingTorch = B._backend == "torch"

    path_data = '/home/flhe/Documents/hdf5/'
    filename_data = 'Lorenz63.hdf5'
    filepath_data = path_data + filename_data

    path_esn = '/home/flhe/Documents/hdf5/'      
    filename_esn = 'Lorenz63_esn.hdf5'
    filepath_esn = path_esn + filename_esn
    ###################################### 
    
    #Study Params 
    randomSeed = range(0,20,1)                                                #random number generator seeds. The no. seeds corresponds to the no. threads that will be launched.
    nseed = len(randomSeed)

    nreservoir_array = np.linspace(500,3000, 10, dtype = int)                 #Here we study the ESN for all combinations of the 10 reservoir sizes
    reservoirDensity_array = np.linspace(0.1,0.4,10)                          #and 10 reservoir densities, i.e. 100 runs.

    study_parameters = (nreservoir_array,reservoirDensity_array)              #tuple specifying the study parameter range. 
                                                                              #Don't forget to specify corresponding ESNParams Setter in line 253
                                                                              #if only one parameter is used the tuple should be (myreservoirparameter_array, )
    nstudyparameters = len(study_parameters)

    #----------------------------------
    #DATA PARAMETERS
    #----------------------------------
    norm = "abs_max"
    data = ImportData(filepath_data)
    data = NormalizeData(data,norm)
    n_input_data = data.shape[1]
    data_timesteps = data.shape[0]                           #no. time steps the orignal data has/should have

    trainingLength = 2000                            #no. time steps for the training data set
    testingLength = 1000                             #no. time stes for the testing/validation data set
    esn_timesteps = trainingLength + testingLength  #no. total resulting time steps for the esn 
    
    #To DO: for now this only works if esn_start != 0
    esn_start = data_timesteps - esn_timesteps      #Index of the original data, at which the training output y_train will begin. 
                                                    #Note that the training input u_train will therefore begin at the index esn_start-1. esn_start therefore must not be 0!
    esn_end = data_timesteps                        #Index of the original data, at which the testing/validation output y_test will end.

    #----------------------------------
    #RESERVOIR PARAMETERS
    #----------------------------------
    
    n_input = n_input_data                           #input data dimensions
    n_output =  n_input_data                         #output data dimensions
    n_reservoir = 2100                               #dimensions of reservoir state and W with shape (n_reservoir, n_reservoir)
    leakingRate = 0.95                               #factor controlling the leaky integrator formulation (1 -> fully nonlinear, 0 -> fully linear)
    spectralRadius = 0.95                            #maximum absolute eigenvalue of W
    regressionParameters = [5e-2]                    #ridge regression/ penalty parameter of ridge regression
    reservoirDensity = 0.2                           #fraction of non-zero elements of W
    noiseLevel = 0#1e-6                              #amplitude of the gaussian noise term in the activation function
    inputScaling = 1.0                               #
    inputDensity = 1.0                               #fraction of non-zero elements of Win
    solver = 'lsqr'                                  #method the training/fitting procedure should use to compute Wout
    weightGeneration = 'naive'                       #method the random weights Win, W should be initialized.
    feedback = False                                 #
    bias = 1.0                                       #input bias in the input mapping: Win*[1;u]
    outputBias = 1.0                                 #output bias in the final output mapping:  y = Wout*[outputbias; outputInputScaling*u; s]
    outputInputScaling = 1.0                         #factor by which the input data should be scaled by in the final output mapping: y = Wout*[outputbias; outputInputScaling*u; s]
    transientTime = "Auto"                           #
    transientTimeCalculationLength = 20              #
    transientTimeCalculationEpsilon = 1e-3           #
    p = 1                                            #Prediction time steps. For now: only p=1 is valid. TO DO: implement more general case.
    ################
    
    #Construct training & testing/validation data from original data.
    u_train, y_train, u_test, y_test = PreparePredictorData(data, n_input, n_output, trainingLength, testingLength, esn_start, esn_end)
               
    #Compute the total number of ESN configurations per seed/ thread
    nstudies = 1
    for iparam in range(nstudyparameters):
        nstudies *= len(study_parameters[iparam])
    print('Seeds: {0}. Studies per seed: {1}'.format(nseed, nstudies))

    #----------------------------------
    #PROCESS ESNPARAMS OBJECTS
    #----------------------------------

    PARAM_OBJ = [ESNParams(randomSeed[iseed],
                esn_start, esn_end,
                trainingLength, testingLength,
                data_timesteps,
                n_input,n_output,n_reservoir,
                leakingRate, spectralRadius,
                reservoirDensity, regressionParameters,
                bias,outputBias, outputInputScaling,
                inputScaling, inputDensity, noiseLevel,
                weightGeneration, solver, feedback,
                p, transientTime, transientTimeCalculationEpsilon,transientTimeCalculationLength)
                for iseed in range(nseed)]


    for obj in PARAM_OBJ:
        obj.SetTrainingData(u_train, y_train)
        obj.SetTestingData(y_test, y_train[-1,:])

        #Convert all numpy arrays of the ESNParams object to torch tensors
        if usingTorch:
            ConvertNumpyToTorch(obj)               

    #Save model parameters to hdf5 file
    print('Saving to Hdf5 file {0}'.format(filepath_esn))
    SaveESNParams(PARAM_OBJ[0],filepath_esn)
        
    #--------------------------------------
    #Start ESN study with nseed threads
    #---------------------------------------
    Threads = [ESNThread(randomSeed[ii], 
                        PARAM_OBJ[ii], 
                        filepath_esn, 
                        study_parameters, 
                        nstudies, 
                        (PARAM_OBJ[ii].SetNReservoir, PARAM_OBJ[ii].SetReservoirDensity),              #<-- specify ESNParams setter functions for corresponding study parameters
                        usingTorch)
                for ii in range(nseed)]


    time_start = time.time()
    for thread in Threads:
        thread.start()

    for thread in Threads:
        thread.join()
    time_end = time.time()


    print('\n ----------------------------------------')
    print('\nESN STUDIES SUCCESFULL!')
    print('\n ----------------------------------------')
    print('\n Total elapsed time {0}'.format(time_end-time_start))







