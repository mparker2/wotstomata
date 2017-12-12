#!/bin/bash

#$ -l mem=26G
#$ -l rmem=26G
#$ -l gpu=1
#$ -m bea
#$ -M mparker2@sheffield.ac.uk
#$ -j y
#$ -N predict

module load apps/python/conda
module load libs/cudnn/5.1/binary-cuda-8.0.44
module load libs/CUDA/8.0.44/binary
source activate sharc_ml

KERAS_BACKEND=tensorflow python /fastdata/mbp14mtp/stomatal_prediction/predict_image.py \
  --img $1 \
  --arch /fastdata/mbp14mtp/stomatal_prediction/segmentation_model_architecture.json \
  --weights /fastdata/mbp14mtp/stomatal_prediction/segmentation_model_weights.h5 \
  --output "${1%%.tif}.pred.h5
