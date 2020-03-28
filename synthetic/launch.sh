#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=8192
#SBATCH --account=rkozma
#SBATCH --output=output/job_%j.log

seed=$1
dataset=$2

python3 gan.py $seed $dataset