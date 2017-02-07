GPUexplore
==========
GPUexplore is a model checker implemented in CUDA. At the moment, it can be used to check for deadlocks as well as safety properties. Input files should be in the gpf-format (more on that below).

GPUexplore supports the following commandline options:
```
GPUexplore <model> [-b <num_blocks>] [-t <num_threads>] [-k <kernel_iter>] [-q <hash_table_size>] [-v <num>] [--por [--cycle-proviso]] [-d|-p]
Generate the state space of <model>.gpf (without extension), using the options specified:
-b                number of blocks to use
-t                number of threads per block
-k                number of iterations per kernel launch
-q                size of the hash table in number of 32 bit integers
-v                verbosity: 0 is quiet, 1 prints the iterations, 2 prints the number of states, 3 prints state vectors
--por             apply partial-order reduction
--cycle-proviso   apply the cycle proviso during POR
-d                check for deadlocks
-p                check for safety property (should be embedded in the model)
```

Input models
------------
The input models in gpf format can be generated from EXP models. Several examples can be found [here](http://tilde.snt.utwente.nl/~thomas.neele/GPUexplore-models.tar.gz). Included is a python script `GPUnetltsgen.py` to perform this conversion. The `-p` option should be used if you later want to apply POR during state-space exploration.

Branches
--------
Since it is very hard to merge the three different implementations of partial-order reduction without duplicating code, there are three separate branches. `ample-por` contains a POR implementation based on the ample-set approach, `cample-por` uses the cluster-based ample-set approach and `stubborn-por` computes the reduction based on stubborn sets.

Publications
--------
Anton Wijs, Thomas Neele, Dragan Bosnacki:
GPUexplore 2.0: Unleashing GPU Explicit-State Model Checking. FM 2016: 694-701

Thomas Neele, Anton Wijs, Dragan Bosnacki, Jaco van de Pol:
Partial-Order Reduction for GPU Model Checking. ATVA 2016: 357-374

Anton Wijs:
BFS-Based Model Checking of Linear-Time Properties with an Application on GPUs. CAV (2) 2016: 472-493

Anton Wijs, Dragan Bosnacki:
Many-core on-the-fly model checking of safety properties using GPUs. STTT 18(2): 169-185 (2016)

Anton Wijs, Dragan Bosnacki:
GPUexplore: Many-Core On-the-Fly State Space Exploration Using GPUs. TACAS 2014: 233-247


Previous versions
--------
The version of the code that was used for the ATVA2016 paper has been tagged with `atva2016` in all three POR branches. Since then, we have mainly developed the cample-set version. The resulting code was used for the FM2016 publication. This commit has been tagged in the `cample-por` branch.