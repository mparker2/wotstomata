#!/bin/bash

#$ -l mem=32G
#$ -l rmem=32G
#$ -l gpu=1
#$ -m bea
#$ -M mparker2@sheffield.ac.uk
#$ -j y
#$ -N train_hm_seg

module load apps/python/conda
module load libs/cudnn/5.1/binary-cuda-8.0.44
module load libs/CUDA/8.0.44/binary
source activate sharc_ml

MIN_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv | \
          cut -d ' ' -f 1 | \
          tail -8 | \
          sort -rn | \
          head -1)
while [ $MIN_MEM -lt 10000 ]; do
    echo "only $MIN_MEM Mb free currently, waiting 5mins"
    sleep 5m
    MIN_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv | \
              cut -d ' ' -f 1 | \
              tail -8 | \
              sort -rn | \
              head -1)
done


CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free --format=csv | \
                       cut -d ' ' -f 1 | \
                       tail -8 | \
                       nl -v 0 | \
                       sort -rnk2 | \
                       head -1 | \
                       cut -f 1)

echo "using GPU $CUDA_VISIBLE_DEVICES"

export CUDA_VISIBLE_DEVICES
export KERAS_BACKEND=tensorflow

cd /fastdata/mbp14mtp/stomatal_prediction/
python ./train_cell_segmentation.py