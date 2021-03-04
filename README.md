# ESNOverlay
An overlay for the python module `easyesn`. Allows for computing the same ESN configuration for different random generator seeds in parallel, using pythons `threading` module. Further, (sequential) grid searches can be specified. The overlay uses the hdf5 file format (`h5py` module) to store the ESN model, the produced reservoir ouputs, as well as their mean square errors.


## The overlay consists of three python files:
  - `esn_user.py` (user settings & main function)
  - `esn_fun.py` (functions used for pre-/ postprocessing and running the ESN)
  - `esn_params.py` (here the ESNParams class is defined. It is used to pass the standard parameter settings of the ESN around the program)

## How to run a study:
Use the `esn_user.py` file.
Say we want to study the behavior of the ESN which should predict the Lorenz 63 model (data for `dt=1e-2`, and `data_timesteps = 5000` can be found in `example_data/Lorenz63_data.hdf5`).We specify the constant ESN parameters in the overlay and vary the reservoir size and reservoir density/sparsity over a coarse $10\timed 10$ grid. This makes up 100 study configurations in total. Further, we sample 20 random realizations for each of those configurations (for statistical analysis). For this we specify the randomSeed array. The overlay will launch one thread for each seed. Important: you have to specify the ESNParams class member functions SetNReservoir and SetReservoirDensity when launching the threads. The reservoir outputs, their mean square error, as well as the constant model parameters are saved to an hdf5. Where they can be read for postprocessing.

## Further
A small postprocessing script can be found in the IPython Notebook read_study.ipynb. This and other files will be extended in the future.

