#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-03:05:00

###
# Calculates train/val scores for all checkpoints for a given mode and 
# flushes them into a csv file
# NOTE: Can take very long 
###


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

data=$1
model=$2
opt=$3

#echo "Step Top1 Top2 Top3 Top4 Top5" > experiments/${model}/train.csv
echo "Step Top1 Top2 Top3 Top4 Top5" > experiments/${model}/valid.csv
# echo "Step Top1 Top2 Top3 Top4 Top5" >> experiments/${model}/test_accuracy.csv
for ckpt in $(ls experiments/${model}/${model}/${model}_model_*);
do
	#python translate.py -model experiments/${model}/${model}/$ckpt -src data/${data}/train.src -output experiments/${model}/train.out -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
	#accs=$(python score_predictions_ds.py -target data/${data}/train.tgt -predictions experiments/${model}/train.out -beam_size 5 -invalid_smiles -$opt | grep Top-[1-5] | sed 's/%//g' | awk '{print $2}')
	#step=$(echo $ckpt | grep -o [0-9][0-9]000)
	#echo $step $accs >> experiments/${model}/train.csv

	python translate.py -model $ckpt -src data/${data}/valid.src -output experiments/${model}/valid.out -batch_size 128 -replace_unk -gpu 1 -max_length 200 -fast -verbose -n_best 5
	accs=$(python score.py -target data/${data}/valid.tgt -predictions experiments/${model}/valid.out -beam_size 5 -invalid_smiles -mol_format $opt| grep Top-[1-5] | sed 's/%//g' | awk '{print $2}')
	step=$(echo $ckpt | grep -o [0-9][0-9]000)
	echo $step $accs >> experiments/${model}/valid.csv
done

