#!/bin/bash
#$ -j y

module load apps/python/conda
module load libs/cudnn/5.1/binary-cuda-8.0.44
module load libs/CUDA/8.0.44/binary

if [ -z "$CONDA_ENV" ];
then
  CONDA_ENV="root"
fi

source activate $CONDA_ENV

if [ -z "$NUM_GPUS" ];
then
  NUM_GPUS="1"
fi

MIN_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv | \
          cut -d ' ' -f 1 | \
          tail -8 | \
          sort -rn | \
          head -n $NUM_GPUS | \
          tail -n 1)
while [ $MIN_MEM -lt 10000 ]; do
    echo "only $MIN_MEM Mb free currently, waiting 5mins"
    sleep 5m
    MIN_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv | \
              cut -d ' ' -f 1 | \
              tail -8 | \
              sort -rn | \
              head -n $NUM_GPUS | \
              tail -n 1)
done

CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv | \
                       cut -d ' ' -f 1 | \
                       tail -8 | \
                       nl -v 0 | \
                       sort -rnk2 | \
                       head -n $NUM_GPUS | \
                       cut -f 1 | \
                       paste -s -d",")

echo "using GPUs $CUDA_VISIBLE_DEVICES"

export CUDA_VISIBLE_DEVICES
export KERAS_BACKEND=tensorflow

if [ -z "$YAML" ];
then
  YAML="./train.yaml"
fi

ws_train run $YAML
