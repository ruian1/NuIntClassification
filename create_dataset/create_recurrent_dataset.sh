#!/bin/bash
module --force purge
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3/setup.sh`
i3env=/cvmfs/icecube.opensciencegrid.org/py2-v3/RHEL_7_x86_64/metaprojects/simulation/V06-00-00/env-shell.sh

#$i3env python /project/6008051/fuchsgru/NuIntClassification/create_dataset/create_recurrent_dataset.py $SLURM_ARRAY_TASK_ID
$i3env python /project/6008051/fuchsgru/NuIntClassification/create_dataset/create_recurrent_dataset.py 0