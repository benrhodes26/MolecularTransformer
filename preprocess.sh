#!/usr/bin/bash
dataset=$1
python preprocess.py -train_src data/${dataset}/train.src -train_tgt data/${dataset}/train.tgt -valid_src data/${dataset}/valid.src -valid_tgt data/${dataset}/valid.tgt -save_data data/${dataset}/${dataset} -src_seq_length 1000 -tgt_seq_length 1000 -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
