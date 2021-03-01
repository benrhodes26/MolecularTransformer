#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-01:05:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

data=deepSMILES15K
model=deepSMILES15K_L4N256

python translate.py -model models/ds_2_model_step_20000.pt -src data/SMILES15K/test.src -output test_result.txt -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
#python translate.py -model experiments/${model}/${model}/${model}_model_step_50000.pt -src data/${data}/test.src -output experiments/${model}/test.out -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
#python translate.py -model experiments/${model}/${model}/${model}_model_step_50000.pt -src data/${data}/train.src -output experiments/${model}/train.out -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
#python translate.py -model experiments/${model}/${model}/${model}_model_step_50000.pt -src data/${data}/valid.src -output experiments/${model}/valid.out -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
