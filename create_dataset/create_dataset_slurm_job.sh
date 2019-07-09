#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --output=log/create_dataset/data_dragon8_part%a.log  # %N for node name, %j for jobID
#SBATCH --account=rpp-kenclark
#SBATCH --mem=8G

eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
singularity exec --bind /cvmfs --bind /scratch/fuchsgru --bind /scratch/hignight --bind /project/6008051/fuchsgru --bind /project/6008051/hignight --bind /home/fuchsgru --bind /home/hignight --nv /project/6008051/hignight/singularity_images/centos7.img /project/6008051/fuchsgru/NuIntClassification/create_dataset/create_dataset.sh

##singularity exec --bind /cvmfs/icecube.opensciencegrid.org/ --bind /home/fuchsgru/ --bind /project/6008051 /project/6008051/hignight/singularity_images/centos7.img bash /project/6008051/fuchsgru/NuIntClassification/create_dataset.sh
