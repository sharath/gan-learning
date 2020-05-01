#!/usr/bin/env bash
mkdir output

seeds=(1 2)
samplers=(1 2 3 4 5 6 7)
datasets=(2)
objectives=(1 2)

for seed in ${seeds[@]}
do
  for sampler in ${samplers[@]}
  do
    for dataset in ${datasets[@]}
    do
      for objective in ${objectives[@]}
      do
        sbatch launch.sh $seed $sampler $dataset $objective
      done
    done
  done
done