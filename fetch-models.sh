#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-04:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

export TMPDIR=/disk/scratch/${STUDENT_ID}

export DATASET_DIR=${TMPDIR}/datasets
export MODEL_DIR=${TMPDIR}/models


# === # === # === # 
# Set the hyperparameters here
num_neurons=128
num_layers=2
# === # === # === #

dataset=SELFIES15K
model=${dataset}_L${num_layers}N${num_neurons}

DATALOCAL=${PWD}/data/$dataset
DATASCRATCH=${DATASET_DIR}/$dataset
MODELLOCAL=${PWD}/experiments/$model
MODELSCRATCH=${MODEL_DIR}/$model

[ -d $MODELSCRATCH ] && {
    echo "models are here copying..."
    rsync -r $MODELSCRATCH $MODELLOCAL
} || {
    echo "Not here, try other node..."
}



