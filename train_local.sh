#!/bin/bash

num_neurons=$1
num_layers=$2
alpha=$3
dataset=$4
model=${dataset}_L${num_layers}N${num_neurons}_${alpha}

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
