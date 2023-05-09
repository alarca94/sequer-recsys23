#!/bin/sh

datasets='amazon-sports amazon-toys amazon-beauty yelp'
suffix='+_r'
model='sequer'
seeds='1111 24 53 126 675'
count=0
cuda=1
for seed in ${seeds}; do
  for dataset in ${datasets}; do
    if [ ${count} -eq 0 ]
    then
      CUDA_VISIBLE_DEVICES=${cuda} python main.py --model-name ${model} --model-suffix=${suffix} --dataset ${dataset} --fold 0 --seed ${seed} --cuda --log-to-file
    else
      CUDA_VISIBLE_DEVICES=${cuda} python main.py --model-name ${model} --model-suffix=${suffix} --dataset ${dataset} --fold 0 --seed ${seed} --cuda --no-generate --log-to-file
    fi
    wait
  done
  (( count++ ))
done
