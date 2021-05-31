#!/usr/bin/env bash

# ! Directory of this project
export DIR_ROBUST_360_8PA="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $DIR_ROBUST_360_8PA/"setup.bash has been source"

source $DIR_ROBUST_360_8PA/env

# ! Dataset directories
export DIR_MP3D_VO_DATASET=${DIR_DATASETS}/ICRA2021
export DIR_TUM_VI_DATASET=${DIR_DATASETS}/TUM_VI

export PYTHONPATH=$PYTHONPATH:$DIR_ROBUST_360_8PA
