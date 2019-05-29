#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --output=log/recurrent.out  # %N for node name, %j for jobID
#SBATCH --account=rpp-kenclark
#SBATCH --mem=128G

module load cuda cudnn
module load python/3.6
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
source ~/myenv/bin/activate
#pip3 install --upgrade pip
#pip3 install --no-index -r requirements.txt
python3 main.py settings/hd5.json
