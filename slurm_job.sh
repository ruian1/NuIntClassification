#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --output=log/64_64_64_64_64_32_16.out  # %N for node name, %j for jobID
#SBATCH --account=rpp-kenclark
#SBATCH --mem=8G

module load cuda cudnn
module load python/3.6
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
source ~/myenv/bin/activate
#pip3 install --upgrade pip
#pip3 install --no-index -r requirements.txt
python3 train.py ../test_data/data_centered.pkl '([64,64,64,64,64],[32,16,1])'
