!#/bin/bash

#$ -l mem=16G
#$ -l rmem=16G
#$ -l gpu=1
#$ -m bea
#$ -M mparker2@sheffield.ac.uk
#$ -j y
#$ -N train_nn

module load apps/python/conda
module load libs/cudnn/5.1/binary-cuda-8.0.44
module load libs/CUDA/8.0.44/binary
source activate sharc_ml

KERAS_BACKEND=tensorflow python ./train_stomatal_prediction.py
