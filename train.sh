#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}

export DATASET_DIR=${TMPDIR}/datasets
export MODEL_DIR=${TMPDIR}/models


# === # === # === # 
# Set the hyperparameters here
num_neurons=$1
num_layers=$2
alpha=$3
# === # === # === #

# prefix for tensorboard log (accuracy report is not right though)
tboard=moltransform

dataset=$4
model=${dataset}_L${num_layers}N${num_neurons}_${alpha}

DATALOCAL=${PWD}/data/$dataset
DATASCRATCH=${DATASET_DIR}/$dataset
MODELLOCAL=${PWD}/experiments/$model
MODELSCRATCH=${MODEL_DIR}/$model

mkdir -p $DATASCRATCH
mkdir -p $MODELSCRATCH

# copy data
rsync -r $DATALOCAL $DATASCRATCH

# Training

# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

python train.py -data $DATASCRATCH/$dataset/$dataset \
	-save_model $MODELSCRATCH/${model}_model \
	-seed 42 \
	-gpu_ranks 0 \
	-save_checkpoint_steps 1000 \
	-keep_checkpoint 50 \
	-train_steps 50000 \
	-param_init 0 \
	-param_init_glorot \
	-max_generator_batches 32 \
	-batch_size 2048 \
	-batch_type tokens \
	-normalization tokens \
	-max_grad_norm 0 -accum_count 2 \
	-optim adam \
	-adam_beta1 0.9 \
	-adam_beta2 0.998 \
	-decay_method noam \
	-warmup_steps 4000 \
	-learning_rate $alpha \
	-label_smoothing 0 \
	-report_every 1000 \
	-valid_steps 1000 \
	-layers $num_layers \
	-rnn_size $num_neurons \
	-word_vec_size $num_neurons \
	-encoder_type transformer \
	-decoder_type transformer \
	-dropout 0.1 \
	-position_encoding -share_embeddings \
	-global_attention general \
	-global_attention_function softmax \
	-self_attn_type scaled-dot \
	-heads 8 -transformer_ff 2048 \
	-tensorboard -tensorboard_log_dir runs/${tboard}_${1}_${2}_${3}_${4}

# copy results back to local
rsync -r $MODELSCRATCH $MODELLOCAL


