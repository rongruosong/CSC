# coding=utf-8
import torch
import logging
from pathlib import Path
from transformers import BertForMaskedLM, BertConfig
import re
import numpy as np

import warnings
from model import BertForCSC
from tokenization import BertTokenizer
# warnings.filterwarnings('ignore')

logger = logging.getLogger("csc" + __name__)

def change_tf_key(model_weight_path: str):
    import tensorflow as tf
    print("Converting TensorFlow checkpoint from {}".format(model_weight_path))
    # Load weights from TF model
    tf_state_dict = {}
    init_vars = tf.train.list_variables(model_weight_path)
    for name, shape in init_vars:
        name_list = name.split('/')
        if any(
            n in ["Variable", "adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name_list
        ):
            # print(f"Skipping {'/'.join(name)}")
            continue
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(model_weight_path, name)

        if 'kernel' in name:
            array = np.transpose(array)
        
        # change name
        if 'encoder' in name:
            name_list = re.split(r'[/_]', name)
        else:
            name_list = re.split(r'[/]', name)
        name = '.'.join(name_list[1:])
        
        if 'embeddings' in name and 'LayerNorm' not in name:
            name += '.weight'
        if name == 'output_weights':
            name = 'classifier.weight'
        if name == 'output_bias':
            name = 'classifier.bias'
        
        for old, new in [['kernel', 'weight'], ['gamma', 'weight'], ['beta', 'bias']]:
            name = name.replace(old, new)
        if name == 'classifier.weight':
            print(array)
        tf_state_dict[name] = array
    return tf_state_dict

def load_tf_cbert(model: torch.nn.Module, model_weight_path: Path):
    import tensorflow as tf
    state_dict = model.state_dict()
    model_weight_path = str(model_weight_path)
    tf_state_dict = change_tf_key(model_weight_path)
    for key in state_dict.keys():
        if key in tf_state_dict.keys():
            state_dict[key] = torch.from_numpy(tf_state_dict[key])
    model.load_state_dict(state_dict, strict=False)
    return model

if __name__ == '__main__':
    config = BertConfig.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer('./cbert/vocab.txt')
    model = BertForCSC(config, len(tokenizer.vocab), -100)
    model = load_tf_cbert(model, Path('../cbert/bert_model.ckpt'))
    print('test:')
    state_dict = model.state_dict()
    print(state_dict['classifier.weight'])

