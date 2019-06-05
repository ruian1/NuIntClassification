#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --output=log/create_recurrent_dataset.log  # %N for node name, %j for jobID
#SBATCH --account=rpp-kenclark
#SBATCH --mem=8G

singularity exec --bind /cvmfs/icecube.opensciencegrid.org/ --bind /home/fuchsgru/ --bind /project/6008051 /project/6008051/hignight/singularity_images/centos7.img bash /project/6008051/fuchsgru/NuIntClassification/create_recurrent_dataset.sh
