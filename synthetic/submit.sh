#!/usr/bin/env bash
mkdir output

datasets=(0 1)
seeds=(0 1 2 3 4)

for d in ${datasets[@]}
do
  for s in ${seeds[@]}
  do
    sbatch launch.sh $s $d
  done
done