# ESNOverlay
An overlay for the python module easyesn. Allows for computing the same ESN configuration for different random generator seeds in parallel, using pythons threading module. Further, (sequential) grid searches can be specified. The overlay uses the hdf5 file format (h5py module) to store the ESN model, the produced reservoir ouputs, as well as their mean square errors.
