#!/bin/bash
module --force purge
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
i3env=/home/hignight/work/oscNext_official/oscNext/build_trunk_jan21_py2_v3.1.1/env-shell.sh

#$i3env python /project/6008051/fuchsgru/NuIntClassification/create_dataset/create_dataset.py $SLURM_ARRAY_TASK_ID
$i3env python /project/6008051/fuchsgru/NuIntClassification/create_dataset/create_dataset.py 0

