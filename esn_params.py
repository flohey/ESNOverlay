class ESNParams:
    def __init__(self, randomSeed,
                esn_start, esn_end,
                trainingLength, testingLength,
                data_timesteps,
                n_input,n_output,n_reservoir,
                leakingRate, spectralRadius,
                reservoirDensity, regressionParameters,
                bias,outputBias, outputInputScaling,
                inputScaling, inputDensity, noiseLevel,
                weightGeneration, solver, feedback,
                p, transientTime, transientTimeCalculationEpsilon,transientTimeCalculationLength):

        self.randomSeed = randomSeed             #RNG seed (numpy)
        self.SeedID = randomSeed                 #RNG seed (torch) used for printing and saving as well

        self.esn_start = esn_start                #To DO: for now this only works if esn_start != 0
        self.esn_end = esn_end
        self.trainingLength = trainingLength
        self.testingLength = testingLength
        self.data_timesteps = data_timesteps
        self.esn_timesteps = trainingLength + testingLength
        self.n_input = n_input                              
        self.n_output = n_output                            
        self.n_reservoir = n_reservoir                      
        self.srows = int(1+n_reservoir+n_input)             #dim of state matrix: (srows,timesteps). srows = bias + reservoir nodes + n_input
        self.leakingRate = leakingRate                      
        self.spectralRadius = spectralRadius
        self.reservoirDensity = reservoirDensity
        self.regressionParameters = regressionParameters
        self.bias = bias
        self.outputBias = outputBias
        self.outputInputScaling = outputInputScaling
        self.inputScaling = inputScaling
        self.inputDensity = inputDensity
        self.noiseLevel = noiseLevel
        self.weightGeneration = weightGeneration
        self.solver = solver
        self.feedback = feedback
        self.p = p                                          #To DO:  We only consider p = 1 for now. The more general case is not yet implemented.

        self.transientTime = transientTime
        self.transientTimeCalculationEpsilon = transientTimeCalculationEpsilon
        self.transientTimeCalculationLength = transientTimeCalculationLength
        
        self.y_train = None
        self.u_train = None
        self.y_test = None
        self.u_test = None
        self.pred_init_input = None


    #----------------------------------
    #FUNCTIONS
    #----------------------------------

    def SetTrainingData(self, u_train, y_train):
        self.u_train = u_train
        self.y_train = y_train

    def SetTestingData(self, y_test, pred_init_input=None):          #TO DO: set u_test. Used when Reservoir = Dynamics Emulator
        
        self.y_test = y_test
        
        if self.y_train is not None and pred_init_input is None:
            #Initial input is last training input. Then first prediction aligns with the first entry of y_test
            self.pred_init_input = self.y_train[-1,:]    #initial input the trained ESN receives for the beginning of the testing phase
        else:
            self.pred_init_input = pred_init_input


    #FH 01/02/2021: Added Setter Functions
    def SetNReservoir(self,n_reservoir, study_dict):
        self.n_reservoir = n_reservoir
        self.srows = int(1+self.n_reservoir+self.n_input)           #adjust srows as according to changed n_reservoir
        study_dict['n_reservoir'] = n_reservoir

    def SetSpectralRadius(self, spectralRadius, study_dict):
        self.spectralRadius = spectralRadius
        study_dict['spectralRadius'] = spectralRadius

    def SetReservoirDensity(self, reservoirDensity, study_dict):
        self.reservoirDensity = reservoirDensity
        study_dict['reservoirDensity'] = reservoirDensity

    def SetLeakingRate(self, leakingRate, study_dict):
        self.leakingRate = leakingRate
        study_dict['leakingRate'] = leakingRate

    def SetRegressionParameters(self, regressionParameters, study_dict):
        self.regressionParameters = regressionParameters
        study_dict['regressionParameters'] = regressionParameters

    def SetInputScaling(self, inputScaling, study_dict):
        self.inputScaling = inputScaling
        study_dict['inputScaling'] = inputScaling
