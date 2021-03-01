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

data=SMILES15K
model=SMILES15K_L2N128
# python translate.py -model models/MIT_mixed_augm_model_average_20.pt -src data/MIT_mixed_augm/src-test.txt -output results.txt -batch_size 64 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
# python translate.py -model experiments/MIT_mixed_augm_L2N128/MIT_mixed_augm_L2N128/MIT_mixed_augm_L2N128/MIT_mixed_augm_L2N128_model_step_100000.pt -src data/MIT_mixed_augm/src-test.txt -output simple_preds.txt -batch_size 256 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
# python translate.py -model experiments/MIT_mixed_augm_L4N256/MIT_mixed_augm_L4N256/MIT_mixed_augm_L4N256/MIT_mixed_augm_L4N256_model_step_20000.pt -src data/MIT_mixed_augm/src-test.txt -output orig_preds.txt -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
python translate.py -model experiments/${model}/${model}/${model}/${model}_model_step_50000.pt -src data/${data}/sm-src-test.txt -output ${model}-tst-out.txt -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
python translate.py -model experiments/${model}/${model}/${model}/${model}_model_step_50000.pt -src data/${data}/sm-src-valid.txt -output ${model}-valid-out.txt -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
