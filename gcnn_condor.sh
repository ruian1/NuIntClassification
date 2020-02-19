#!/bin/bash

eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.0.1/setup.sh`
source ~/myenv/bin/activate
cd ~/NuIntClassification
python3 train.py settings/hd5.json

