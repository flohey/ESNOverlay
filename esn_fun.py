#easyesn backends B which are stable for this code (change in .easyesn/easyesn.json)
import numpy as np
import torch

#save data structure
import h5py 

#easyesn esn
import easyesn
from easyesn import PredictionESN, BaseESN
from easyesn import backend as B

from esn_params import ESNParams



###########################################################################################################

#                             PRE-PROCESSING/ IMPORTING

###########################################################################################################

#--------------------------------------------------------------------------
def ImportData(filepath_data):
    '''Import the data specified in the hdf5 file filepath_data. This function may be adapted to the means of the user. 
       The data should have the shape (timesteps, modes)

        INPUT:
            filepat_data - path where the data file is located

        OUTPUT:
            data - the data set of shape (timesteps, modes)   
    '''

    with h5py.File(filepath_data,'r') as f:
        data = np.array(f.get('data'))

    return data
#--------------------------------------------------------------------------
def NormalizeData(data, norm="abs_max"):
    '''Normalizes data to range [-1,1], by subtracting the mean value of each mode 
       and normalizing by the largest absolute value or standard deviation of each mode.
    
        INPUT:
            data - data that should be normalized for further processing
            norm - string, specifying which norm should be used. 
                abs_max - normalization w.r.t. maximum absolute value of each mode 
                std     - normalization w.r.t. standard deviation of each mode 
                
        RETURN:
            data_norm - data normalized to [-1,1]
    '''
    
    data_norm = data - np.mean(data, axis = 0)
    
    if norm == "std":
        std_data = np.std(data, axis = 0)
        data_norm /= std_data
    
    if norm == "abs_max":
        max_data = np.max(np.abs(data_norm), axis = 0 )
        data_norm /=max_data
        
    return data_norm
#--------------------------------------------------------------------------
def PreparePredictorData(data, n_input, n_output, trainingLength, testingLength, esn_start, esn_end):
    ''' Prepares the input and output training and testing/validation data set for the autonomous predictor ESN case, i.e. where in the
        testing pase, the output data is fed back into the reservoir input layer.
        The data set u_test is therefore not needed, as there is only one initial input in the testing phase.
        Note that n_output = n_input.

    INPUT:
        data           - data array which is supposed to be used for the ESN. The shape must be (data_timesteps, nmodes)
        n_input        - input dimensions/ no. input modes
        n_output       - output dimensions/ no. output modes
        trainingLength - no. time steps for the training data set
        testingLength  - no. time steps for the testing/validation data set
        esn_start      - Index of the original data, at which the training output y_train will begin. 
        esn_end        - Index of the original data, at which the testing/validation output y_test will end.

    RETURN:
        u_train - training input data set. Shape: (trainingLength, n_input)
        y_train - training output data set. Shape: (trainingLength, n_output)
        u_test  - testing/validation input data set (not used in auotnomous predictor mode). Shape: (testingLength, n_input)
        y_test  - testing/validation output data set. Shape: (testingLength, n_output)
    '''
    
    data_esn = data[esn_start-1:esn_end,:n_input]
    u_train = data_esn[0:trainingLength,:]
    y_train = data_esn[1:trainingLength+1,:]
    
    u_test = data_esn[trainingLength:trainingLength+testingLength,:]
    y_test = data_esn[trainingLength+1:,:]

    return u_train, y_train, u_test, y_test
#--------------------------------------------------------------------------

###########################################################################################################

#                            RUNNING AN ESN

###########################################################################################################

#--------------------------------------------------------------------------
def PredictModel(esn, esn_params, s_fit):
    ''' 
    Predicts values for the next time steps, starting from initial input pred_input. 
    The reservoir state is set to the last state of the training phase. 
    Then the reservoir output is re-routed to the input layer (autonomous predictior)

    INPUT: 
        esn        - PredictionESN class object, containing reservoir data (weight matrices, states, ...).
        esn_params - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.
        s_fit      - state matrix of the training phase.

    RETURN:
        y_pred     - reseroir outputs, produced by the given reservoir specified in esn_params.
        s_pred     - reservoir state matrix of the testing/prediction phase.
    '''
    #Check whether initial input data is available
    if esn_params.pred_init_input is None:
        print('Error in PredictModel: Initial prediction input is not defined!')
        exit()

    #Reset reservoir state to end of training phase:
    esn._x = B.array(s_fit[int(1+esn_params.n_input):esn_params.srows,-1]).reshape(esn_params.n_reservoir,1)   #set state s: last training state (should be satisfied automatically here)
    y_pred = B.zeros([esn_params.n_input,esn_params.testingLength]) 
    s_pred = B.zeros([esn_params.srows, esn_params.testingLength])
    pred_input = esn_params.pred_init_input

    for it in range(esn_params.testingLength):
        s = esn.propagate(pred_input.reshape(1,esn_params.n_input))      #state at time it  (srows,1)                                            
        pred_output = B.array(esn._WOut@s)                         #prediction/reservoir output at time it    

        y_pred[:,it] = pred_output.reshape(esn_params.n_input,)
        s_pred[:,it] = s.reshape(esn_params.srows,)
        
        pred_input = pred_output                             #new input is the current output (autonomous predictions)

    y_pred = y_pred.T                #stay consistent with axis order: (timesteps, modes)
        
    return y_pred, s_pred

#--------------------------------------------------------------------------------------------------------
def BuildAndFitModel(esn_params):
    '''
    Build the reservoir with the in esn_params specified parameters. 
    After that the model is fitted/trained with the training data specified in esn_params.

    INPUT:
        esn_params - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.

    OUTPUT:
        esn       - PredictionESN class object, containing reservoir data (weight matrices, states, ...).
        mse_train - mean square error of (teacher forced!) reservoir output (to training target data set y_train) in the training phase. Mean w.r.t. timestep- & mode-axis. 

    '''
    if esn_params.SeedID == 0:
        #suppress output for other threads
        print("Thread {0}: Building ESN Model".format(esn_params.SeedID))

    esn = PredictionESN(n_input = esn_params.n_input, 
                    inputDensity = esn_params.inputDensity,
                    n_output = esn_params.n_output, 
                    n_reservoir = esn_params.n_reservoir, 
                    leakingRate = esn_params.leakingRate, 
                    spectralRadius = esn_params.spectralRadius, 
                    regressionParameters = esn_params.regressionParameters, 
                    solver = esn_params.solver, 
                    weightGeneration = esn_params.weightGeneration,
                    inputScaling = esn_params.inputScaling, 
                    reservoirDensity = esn_params.reservoirDensity, 
                    randomSeed = esn_params.randomSeed,
                    feedback = esn_params.feedback, 
                    noiseLevel = esn_params.noiseLevel,
                    bias = esn_params.bias,
                    outputBias = esn_params.outputBias,
                    outputInputScaling = esn_params.outputInputScaling)
    
    
    if esn_params.SeedID == 0:
        #suppress output for other threads
        print("Thread {0}: Fitting ESN Model\n".format(esn_params.SeedID)) 



    try:
        rmse_train = esn.fit(esn_params.u_train,
                         esn_params.y_train, 
                         transientTime = esn_params.transientTime, 
                         verbose =0, 
                         transientTimeCalculationLength = esn_params.transientTimeCalculationLength, 
                         transientTimeCalculationEpsilon = esn_params.transientTimeCalculationEpsilon)

        mse_train = rmse_train**2

    except:
        #e.g. transient time too long
        mse_train = None

    finally:
        

        return esn, mse_train
#--------------------------------------------------------------------------
def ConvertNumpyToTorch(esn_params):
    ''' Casts all members of esn_params that are numpy instances to python built in types and torch.tensor.
        Useful when applying data of easyesn with numpy backend to easyesn run with torch backend.
        
        INPUT: 
            esn_params - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.
 '''
    
    allmembers = [m for m in dir(esn_params) if m[:2] != '__' and m[-2:] != '__']
    ifaddmember = [not callable(getattr(esn_params, name)) for name in allmembers ]
    
    #Cast from numpy to python/ torch types
    ii = 0
    for member in allmembers:
        if ifaddmember[ii]:
            value = getattr(esn_params,member)
            
            if isinstance(value, np.float64):           #np.float -> float
                new_value = float(value)      
            elif isinstance(value, np.int64):           #np.int   -> int
                new_value = int(value) 
            elif isinstance(value, np.ndarray):         #array    -> tensor
                new_value = torch.from_numpy(value)   
            else:
                new_value = value
            
            setattr(esn_params, member, new_value)
            
        ii += 1
#--------------------------------------------------------------------------
def ComputeMSE(y_test, y_pred):
    '''
    Computes the mean square error between target data y_test and prediction data y_pred.

    INPUT:
        y_test - validation/testing/ true output
        y_pred - reseroir outputs

    OUTPUT:
        Mean square error  between y_test and y_pred w.r.t. both timestep- & mode-axis.
    '''

    return B.mean((y_test-y_pred)**2, axis = (0,1))
#--------------------------------------------------------------------------
def RunESNWithParams(esn_params):
    ''' Runs the Echo State Network (ESN)/ reservoir with the specified parameters, training and validation/testing data specfied in esn_params.
    
        INPUT: 
            esn_params - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.

        RETURN:
            mse_train - mean square error of (teacher forced!) reservoir output (to training target data set y_train) in the training phase. Mean w.r.t. timestep- & mode-axis. 
            mse_test  - mean square error of reservoir output (to validation data set y_test). Mean w.r.t. timestep- & mode-axis. 
            y_pred    - reseroir outputs, produced by the given reservoir specified in esn_params.
    '''
    
   
    esn, mse_train = BuildAndFitModel(esn_params)

    if mse_train is None:
        #an error occured
        print("While fitting the model, an error occured. Assuming default values.")
        mse_train = False
        mse_test = False
        y_pred = B.zeros([esn_params.testingLength, esn_params.n_output])

        return mse_train, mse_test, y_pred


    #Prediction Phase
    #Win = esn._WInput     #Randomly generated Input Matrix  (set: sparseness/density, scaling of columns)    (srows,n_input)
    #Wres = esn._W         #Randomly generated Reservoir Matrix (set: spectral radius, sparseness/density)    (n_reservoir, n_reservoir)
    #Wout = esn._WOut      #Fitted output matrix, based on input data X_train and target output data y_train. (n_input, srows)
    s_fit = esn._X        #state matrix, contains all states after initial transient time that were propagated based on training input X_train
    
    #esn_params.Win = Win   #uncomment if needed later
    #esn_params.Wres = Wres
    #esn_params.Wout = Wout
    #esn_params.s_fit = s_fit

    if esn_params.SeedID == 0:
        print('Thread {0}: Predicting reservoir outputs'.format(esn_params.SeedID))
    y_pred, s_pred = PredictModel(esn = esn,esn_params = esn_params,s_fit = s_fit)
    esn_params.s_pred = s_pred

    mse_test = ComputeMSE(esn_params.y_test, y_pred)
    
    return mse_train, mse_test, y_pred
#---------------------------------------------------------------------------

###########################################################################################################

#                            SAVING ESN RESULTS

###########################################################################################################
#---------------------------------------------------------------------------
def SaveESNParams(esn_params,filepath):
    ''' Saves the reservoir parameters, training and validation/testing data from ESNParams class object into a hdf5 file
    
        INPUT:
            esn_params - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.
            filepath   - path to which the hdf5 file of the ESN study is saved to
    '''
    
    print('Saving ESNParams')
    with h5py.File(filepath,'w') as f:
        
        
        G_params = f.create_group('ESNParams')
        
        #Model
        G_params.attrs['p'] = esn_params.p
        G_params.attrs['data_timesteps'] = esn_params.data_timesteps
        G_params.attrs['trainingLength'] = esn_params.trainingLength
        G_params.attrs['testingLength'] = esn_params.testingLength
        G_params.attrs['n_input'] = esn_params.n_input
        G_params.attrs['n_output']= esn_params.n_output
        G_params.attrs['n_reservoir'] = esn_params.n_reservoir
        G_params.attrs['leakingRate']= esn_params.leakingRate
        G_params.attrs['spectralRadius']= esn_params.spectralRadius
        G_params.attrs['regressionParameters'] = esn_params.regressionParameters
        G_params.attrs['reservoirDensity'] = esn_params.reservoirDensity
        G_params.attrs['noiseLevel'] = esn_params.noiseLevel         
        G_params.attrs['inputScaling'] = esn_params.inputScaling               
        G_params.attrs['inputDensity'] = esn_params.inputDensity
        G_params.attrs['randomSeed'] = esn_params.randomSeed
        G_params.attrs['solver'] = esn_params.solver
        G_params.attrs['weightGeneration'] = esn_params.weightGeneration

        if esn_params.feedback is None:
            esn_params.feedback = False
        G_params.attrs['feedback'] = esn_params.feedback                       #None raises error
        
        G_params.attrs['bias'] = esn_params.bias
        G_params.attrs['outputBias'] = esn_params.outputBias
        G_params.attrs['outputInputScaling'] = esn_params.outputInputScaling
        G_params.attrs['transientTime'] = esn_params.transientTime
        G_params.attrs['transientTimeCalculationLength'] = esn_params.transientTimeCalculationLength
        G_params.attrs['transientTimeCalculationEpsilon'] = esn_params.transientTimeCalculationEpsilon
        G_params.attrs['esn_start'] = esn_params.esn_start
        G_params.attrs['esn_end'] = esn_params.esn_end
        
        #Datasets
        G_params.create_dataset('y_train',   data = esn_params.y_train, compression = 'gzip', compression_opts = 9)
        G_params.create_dataset('y_test',    data = esn_params.y_test, compression = 'gzip', compression_opts = 9)
        G_params.create_dataset('u_train',   data = esn_params.u_train, compression = 'gzip', compression_opts = 9)
        #G_params.create_dataset('u_test',   data = esn_params.u_train, compression = 'gzip', compression_opts = 9)
        
#--------------------------------------------------------------------------
def SaveModelStudy(filepath, esn_params, y_pred, mse_train, mse_test, StudyID, study_dict):
    '''Saves the ESN parameters from esn_params into a hdf5 file.
       The h5py file has to be init. with ReadModel (saving the fix parameters) before calling this function!
       
       INPUT:
          filepath   - path to which the hdf5 file of the ESN study is saved to
          esn_params - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.
          y_pred     - reseroir outputs, produced by the study/ given reservoir specified in esn_params.
          mse_train  - mean square error of (teacher forced!) reservoir output (to training target data set y_train) in the training phase. Mean w.r.t. timestep- & mode-axis. 
          mse_test   - mean square error of reservoir output (to validation data set y_test). Mean w.r.t. timestep- & mode-axis. 
          StudyID    - ID specifying the study
          study_dict - dictionary specifying the study parameter setting/configuration

          
       '''
        
    if esn_params.randomSeed is not None or esn_params.SeedID == 0:
        #suppress output for other threads
        print('Thread {0}: Saving Model'.format(esn_params.SeedID))
        
        
    with h5py.File(filepath,'a') as f:    
        
        #HDF5-Structure:
        #- SeedID1
        #   - StudyID1
        #       - study_dict
        #       - y_pred
        #       - mse
        
        #   - StudyID2
        #       ...  
        #   - ...
        #- SeedID2
        #   -StudyID1
        #       ...
        #   -StudyID2
        #   - ...
        #- ...
        
        if StudyID == 0:
            G_seed = f.create_group(str(esn_params.SeedID))
        else:
            G_seed = f.get(str(esn_params.SeedID))
            
        G_study = G_seed.create_group(str(StudyID))
        
        for param in study_dict.keys():
            G_study.attrs[param] = study_dict[param]  
        
        G_study.create_dataset('y_pred', data = y_pred, compression = 'gzip', compression_opts = 9)
        G_study.attrs['mse_train'] = mse_train
        G_study.attrs['mse_test'] = mse_test
        
#--------------------------------------------------------------------------

###########################################################################################################

#                            POST-PROCESSING TOOLS

###########################################################################################################
#--------------------------------------------------------------------------
#FH added 04.02.2021
#--------------------------------------------------------------------------
def ReadParallelSeedStudy(filepath, SeedID, study_parameters, nstudy = None):
    '''Imports the results of the ESN study which are saved in a hdf5 file. 
        
        INPUT:
            filepath     - path to which the hdf5 file of the ESN study was saved to
            SeedID       - ID specifying the random number generator (RNG) seed with to which the RNG was set to, when the reservoir was generated. E.g. seedID = 0
            study_parameters - list of strings specifying which parameters were studied. E.g. when reservoir size and density are studied: study_parameters = ['n_reservoir', 'reservoirDensity']
            nstudy       - number of different reservoir setting that were studied. If nstudy = None, the number is deduced from the file.
        RETURN:
            mse_test    - mean square error of reservoir output (to validation data set y_test). Mean w.r.t. timestep- & mode-axis. 
            y_pred      - reseroir outputs, for each study parameter setting of the study 
            study_dicts - dictionary specifying the study parameter setting/configuration  
    '''
        

    with h5py.File(filepath,'r') as f:

        mse_train, mse_test, y_pred, study_dicts = [], [], [], []
        G_seed = f.get(str(SeedID))
        
        #if user does not specify study number, the number is deduced from no. subgroups
        if nstudy is None:
            nstudy = len(G_seed.keys())
        
        for study_id in range(nstudy):
            study_dict = {}
            G_study = G_seed.get(str(study_id))

            mse_train.append(G_study.attrs['mse_train'])
            mse_test.append(G_study.attrs['mse_test'])
            y_pred.append(np.array(G_study.get('y_pred')))

            for name in study_parameters:
                study_dict[name] = G_study.attrs[name]
                
            study_dicts.append(study_dict)
        

    return mse_train, mse_test, y_pred, study_dicts

#--------------------------------------------------------------------------
#FH added 28.02.2021
#--------------------------------------------------------------------------
def ReadESNParams(filepath):
    ''' Reads the ESN parameters from filepath. Creates ESNParams object and returns it.
    
        INPUT:
            filepath - path to which the hdf5 file of the ESN study was saved to

        RETURN:
            esn_params - ESNParams class object containing the reservoir parameters and training & validation/testing data sets.
    '''  

    print('Reading ESNParams.')

    with h5py.File(filepath,'r') as f:

        G_params = f.get('ESNParams')
        trainingLength = G_params.attrs['trainingLength']
        testingLength  = G_params.attrs['testingLength']
        data_timesteps = G_params.attrs['data_timesteps']
        n_input        = G_params.attrs['n_input']
        n_output       = G_params.attrs['n_output']
        n_reservoir    = G_params.attrs['n_reservoir']
        leakingRate    = G_params.attrs['leakingRate']
        spectralRadius  = G_params.attrs['spectralRadius']
        regressionParameters    = G_params.attrs['regressionParameters']
        reservoirDensity        = G_params.attrs['reservoirDensity']
        noiseLevel              = G_params.attrs['noiseLevel']
        inputScaling            = G_params.attrs['inputScaling']
        inputDensity            = G_params.attrs['inputDensity']
        randomSeed              = G_params.attrs['randomSeed']
        solver                  = G_params.attrs['solver']
        weightGeneration        = G_params.attrs['weightGeneration']
        bias                    = G_params.attrs['bias']
        outputBias              = G_params.attrs['outputBias']
        outputInputScaling                 = G_params.attrs['outputInputScaling']
        transientTime                      = G_params.attrs['transientTime']
        transientTimeCalculationLength     = G_params.attrs['transientTimeCalculationLength']
        transientTimeCalculationEpsilon    = G_params.attrs['transientTimeCalculationEpsilon']
        esn_start    = G_params.attrs['esn_start']
        esn_end      = G_params.attrs['esn_end']
        p            = G_params.attrs['p']
        feedback  = G_params.attrs['feedback']

        if feedback == False:
            feedback = None
        if randomSeed == False:
            randomSeed = None

        y_train = np.array(G_params.get('y_train'))
        u_train = np.array(G_params.get('u_train'))
        y_test = np.array(G_params.get('y_test'))
        u_test = None                                 #TO DO: implement u_test
  
            

    esn_params = ESNParams(randomSeed,
                esn_start, esn_end,
                trainingLength, testingLength,
                data_timesteps,
                n_input,n_output,n_reservoir,
                leakingRate, spectralRadius,
                reservoirDensity, regressionParameters,
                bias,outputBias, outputInputScaling,
                inputScaling, inputDensity, noiseLevel,
                weightGeneration, solver, feedback,
                p, transientTime,transientTimeCalculationEpsilon,transientTimeCalculationLength)
    esn_params.SetTrainingData(u_train, y_train)
    esn_params.SetTestingData(y_test)
    
    return esn_params
#--------------------------------------------------------------------------
# FH added 28.02.2021
#--------------------------------------------------------------------------
def CreateStudyConfigArray(study_parameters, study_dicts):
    ''' Computes an array, which gives the parameter configuration/setting for the corresponding study.
        
        INPUT: 
            study_parameters - list of strings specifying which parameters were studied. E.g. when reservoir size and density are studied  study_parameters = ['n_reservoir', 'reservoirDensity']
            study_dicts  - dictionary specifying the study parameter setting/configuration

        RETURN:
            - config - array indicating the parameter setting for given study
        '''

    nparam = len(study_parameters)      #no. different parameters that are studied
    nstudy = len(study_dicts)    #no. studies/ parameter settings that were conducted
    config = np.empty([nstudy,nparam])

    for ii in range(nstudy):
        config_dict  =study_dicts[ii]
        for pp in range(nparam):
            key = study_parameters[pp]
            config[ii,pp] = config_dict[key]
            
    return config
#--------------------------------------------------------------------------
