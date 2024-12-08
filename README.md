# Distributional RL Decision and Control

This repository provides the code implementation of our RA-L paper.

<p align="center">
<img width="900" height="506" src="AC_IQN_based_system.jpg"> 
</p>

## Build VRX Simulator
The Gazebo based simulator [VRX](https://github.com/osrf/vrx) is used for simulation experiments. We developed new packages that realize the navigation system and added them to the original simulator. The simulation envrionment can be built as follows.  

Download LibTorch from [here](https://download.pytorch.org/libtorch/cpu/) and add it to environment path. Our code implementation uses the version 2.2.1+cpu. 
```
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
source ~/.bashrc
sudo ldconfig
```

Set "/path/to/libtorch" to the corresponding location in line 5 of vrx-2.3.2/action_planner/CMakeLists.txt:
```
list(APPEND CMAKE_PREFIX_PATH "/path/to/libtorch")
```

Navigate to the root directory of this repo and run the following commands
```
mkdir -p vrx_ws/src
cp -r vrx-2.3.2/* vrx_ws/src
cd vrx_ws
source /opt/ros/humble/setup.bash
colcon build --merge-install
. install/setup.bash
cp src/run_vrx_experiments.py .
```