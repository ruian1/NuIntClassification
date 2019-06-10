#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --output=log/concatenate_datasets.out  # %N for node name, %j for jobID
#SBATCH --account=rpp-kenclark
#SBATCH --mem=16G

#module load cuda cudnn
module load python/3.6
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
source ~/myenv/bin/activate
#pip3 install --upgrade pip
#pip3 install --no-index -r requirements.txt
#python3 main.py settings/hd5.json
python3 concatenate_datasets.py "../data/data_dragon4_parts/*.hd5" ../data/data_dragon4.hd5
