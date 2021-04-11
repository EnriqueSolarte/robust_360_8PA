#!/usr/bin/env bash

# ! Directory of this project
export DIR_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $DIR_ROOT/"setup.bash has been source"
# ! Dataset directories
export DIR_MP3D_VO_DATASET=${HOME}/Documents/datasets/MP3D_VO
export DIR_TUM_VI_DATASET=${HOME}/Documents/datasets/TUM_VI

export PYTHONPATH=$PYTHONPATH:$DIR_ROOT
