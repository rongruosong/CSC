#!/bin/sh
CURRENT_DIR=`pwd`

train_data_path=$CURRENT_DIR/data/sighan_2015/train.tsv
test_data_path=$CURRENT_DIR/data/sighan_2015/test.tsv

config_path=$CURRENT_DIR/cbert/bert_config.json
vocab_path=$CURRENT_DIR/cbert/vocab.txt

pretrained_model_path=$CURRENT_DIR/cbert/pytorch_model.bin
output_model_path=$CURRENT_DIR/finetune_checkpoint

confusions=$CURRENT_DIR/confusions

train_batch_size=32
test_batch_size=32

seq_length=128

accelerator="GPU"
num_devices=1
num_nodes=1
strategy="ddp"

num_epochs=3

val_check_interval=10

report_steps=50

ignore_index=-100
seed=42

python3 run_finetuning.py \
    --train_data_path $train_data_path \
    --test_data_path $test_data_path \
    --config_path $config_path \
    --vocab_path $vocab_path \
    --pretrained_model_path $pretrained_model_path \
    --output_model_path $output_model_path \
    --confusions $confusions \
    --train_batch_size $train_batch_size \
    --test_batch_size $test_batch_size \
    --seq_length $seq_length \
    --num_devices $num_devices \
    --num_nodes $num_nodes \
    --strategy $strategy \
    --num_epochs $num_epochs \
    --val_check_interval $val_check_interval \
    --ignore_index $ignore_index \
    --seed $seed