# Installation
(Tested on Ubuntu 20.04)

### Overview

1. Set up a Conda env.
2. Install python modules (including bundled dependencies).


### Set up Conda environment
After installing [Anaconda](https://docs.anaconda.com/anaconda/install/), enter the repo directory and create the new environment:
```shell script
conda env create -f environment.yml
conda activate rlapps
bash ./install.sh
```

### Install Python modules

#### 1. DeepMind OpenSpiel (included dependency)
DeepMind's [OpenSpiel](https://github.com/deepmind/open_spiel) is used for poker game logic as well as tabular game utilities.
We include a slightly modified [fork](https://github.com/indylab/open_spiel) as a dependency.

With the new conda env active:
```shell script
# Starting from the repo root
cd dependencies/open_spiel
./install.sh # git clones and apt installs additional dependencies
pip install -e . # This will start a compilation process. May take a few minutes.
cd ../..
```

#### 2. The GRL Package (main package).
```shell script
# Starting from the repo root
pip install -e .
```

Installation is now done!

### Advanced Installation Notes (Optional)

If you need to compile/recompile OpenSpiel without pip installing it, perform the following steps with your conda env *active*. (The conda env needs to be active so that OpenSpiel can find and compile against the python development headers in the env. Python version related issues may occur otherwise):
```shell script
mkdir build
cd build
CC=clang CXX=clang++ cmake -DPython_TARGET_VERSION=3.6 -DCMAKE_CXX_COMPILER=${CXX} -DPython3_FIND_VIRTUALENV=FIRST -DPython3_FIND_STRATEGY=LOCATION ../open_spiel
make -j$(nproc)
cd ../../..
```

To import OpenSpiel without using pip, add OpenSpiel directories to your PYTHONPATH in your ~/.bashrc ([more details here](https://github.com/deepmind/open_spiel/blob/244d1b55eb3f9de2ab4a0e06341ff2847afea466/docs/install.md)):
```shell script
# Add the following lines to your ~/.bashrc:
# For the python modules in open_spiel.
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>
# For the Python bindings of Pyspiel
export PYTHONPATH=$PYTHONPATH:/<path_to_open_spiel_submodule>/build/python
```

### Next Steps

See [Running Experiments](/docs/experiments.md)

