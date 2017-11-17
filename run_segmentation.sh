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

cd /fastdata/mbp14mtp/stomatal_prediction/
KERAS_BACKEND=tensorflow python ./train_cell_segmentation.py
