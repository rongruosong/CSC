#!/bin/sh
CURRENT_DIR=`pwd`

train_data_path=$CURRENT_DIR/data/sighan_2015/train.tsv
test_data_path=$CURRENT_DIR/data/sighan_2015/test.tsv

config_path=$CURRENT_DIR/cbert/bert_config.json
vocab_path=$CURRENT_DIR/cbert/vocab.txt

pretrained_model_path=$CURRENT_DIR/cbert/pytorch_model.bin
output_model_path=$CURRENT_DIR/