#!/usr/bin/env bash
mkdir output

seeds=(0 1 2 3 4 5 6 7 8 9)
experiments=(2 3 4 5 6 7 8)

for s in ${seeds[@]}
do
  for e in ${experiments[@]}
  do
      sbatch launch.sh $s $e
  done
done