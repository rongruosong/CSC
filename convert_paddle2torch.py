# coding=utf-8
import os

import paddle
import torch
from transformers import BertConfig

from ernie import ErnieForMaskedLM

from paddlenlp.transformers import ErnieTokenizer, ErnieForPretraining
import numpy as np


def change_paddle_key():
    paddle_state_dict = {}

    # embedding
    paddle_state_dict['ernie.embeddings.word_embeddings.weight'] = 'ernie.embeddings.word_embeddings.weight'
    paddle_state_dict['ernie.embeddings.position_embeddings.weight'] = 'ernie.embeddings.position_embeddings.weight'
    paddle_state_dict['ernie.embeddings.token_type_embeddings.weight'] = 'ernie.embeddings.token_type_embeddings.weight'
    paddle_state_dict['ernie.embeddings.LayerNorm.weight'] = 'ernie.embeddings.layer_norm.weight'
    paddle_state_dict['ernie.embeddings.LayerNorm.bias'] = 'ernie.embeddings.layer_norm.bias'
    paddle_state_dict['ernie.embeddings.task_type_embeddings.weight'] = 'ernie.embeddings.task_type_embeddings.weight'

    # encoder
    for i in range(12):
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.query.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.q_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.query.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.q_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.key.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.k_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.key.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.k_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.value.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.v_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.self.value.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.v_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.dense.weight'.format(i)] = 'ernie.encoder.layers.{}.self_attn.out_proj.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.dense.bias'.format(i)] = 'ernie.encoder.layers.{}.self_attn.out_proj.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.LayerNorm.weight'.format(i)] = 'ernie.encoder.layers.{}.norm1.weight'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.attention.output.LayerNorm.bias'.format(i)] = 'ernie.encoder.layers.{}.norm1.bias'.format(i)
        paddle_state_dict['ernie.encoder.layer.{}.intermediate.dense.weight'.format(i)] = 'ernie.encoder.layers.{}.linear1.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.intermediate.dense.bias'.format(i)] = 'ernie.encoder.layers.{}.linear1.bias'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.dense.weight'.format(i)] = 'ernie.encoder.layers.{}.linear2.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.dense.bias'.format(i)] = 'ernie.encoder.layers.{}.linear2.bias'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.LayerNorm.weight'.format(i)] = 'ernie.encoder.layers.{}.norm2.weight'.format(i) 
        paddle_state_dict['ernie.encoder.layer.{}.output.LayerNorm.bias'.format(i)] = 'ernie.encoder.layers.{}.norm2.bias'.format(i) 

    paddle_state_dict['ernie.pooler.dense.weight'] = 'ernie.pooler.dense.weight'
    paddle_state_dict['ernie.pooler.dense.bias'] = 'ernie.pooler.dense.bias'
    paddle_state_dict['cls.predictions.bias'] = 'cls.predictions.decoder_bias'
    paddle_state_dict['cls.predictions.transform.dense.weight'] = 'cls.predictions.transform.weight'
    paddle_state_dict['cls.predictions.transform.dense.bias'] = 'cls.predictions.transform.bias'
    paddle_state_dict['cls.predictions.transform.LayerNorm.weight'] = 'cls.predictions.layer_norm.weight'
    paddle_state_dict['cls.predictions.transform.LayerNorm.bias'] = 'cls.predictions.layer_norm.bias'
    paddle_state_dict['cls.predictions.decoder.weight'] = 'ernie.embeddings.word_embeddings.weight'

    return paddle_state_dict

def convert(model, pd_model_weight_path, save_path):
    # 加载paddle参数
    paddle_key_params = paddle.load(pd_model_weight_path)

    paddle_state_dict = change_paddle_key()
    state_dict = model.state_dict()
    for key in state_dict.keys():
        
        if key in paddle_state_dict.keys():
            param = paddle_key_params[paddle_state_dict[key]]
            if 'weight' in key and 'LayerNorm' not in key and 'embeddings' not in key and 'decoder' not in key:
                param = param.transpose((1, 0))
            state_dict[key] = torch.from_numpy(param.numpy())
        else:
            print(key)
    model.load_state_dict(state_dict, strict=False)
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    pd_model_weight_path = '/root/autodl-nas/ernie_3.0_base_zh/ernie_3.0_base_zh.pdparams'
    config_dir = '/root/autodl-nas/ernie_3.0_base_zh/ernie_3.0_base_zh_config.json'
    config = BertConfig.from_json_file(config_dir)
    model = ErnieForMaskedLM(config, -100, True)
    save_path = '/root/autodl-nas/ernie_3.0_base_zh/pytorch_model.bin'
    # convert(model, pd_model_weight_path, save_path)

    # 验证参数转换后， torch的结果是否与paddle保持一致

    tokenizer = ErnieTokenizer.from_pretrained('ernie-3.0-base-zh')
    input = tokenizer.encode('hello world')

    # torch
    model.load_state_dict(torch.load('/root/autodl-nas/ernie_3.0_base_zh/pytorch_model.bin'), strict=False)
    model.eval()
    ids = torch.LongTensor(np.expand_dims(input['input_ids'], 0))
    print(model(ids)[0].detach().numpy())

    # paddle
    """
    ids = paddle.to_tensor(np.expand_dims(input['input_ids'], 0))
    model = ErnieForPretraining.from_pretrained('/root/autodl-nas/ernie_3.0_base_zh/model_state.pdparams')
    model.eval()
    print(model(ids))
    """
