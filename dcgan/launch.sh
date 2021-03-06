#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=16384
#SBATCH --account=rkozma
#SBATCH --output=output/job_%j.log

seed=$1
experiment=$2

seed=$1
sampler=$2
dataset=$3
objective=$4

echo python3 gan.py $seed $sampler $dataset $objective
python3 gan.py $seed $sampler $dataset $objective