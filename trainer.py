# coding=utf-8
"""
该脚本未写完，主要原因在于LightningLite目前无法很好的处理，下面的情况：
    mixed precision 和 grad_clip 同时启用
    deepspeed 与 grad_clip同时也需要定制开发代码
"""
from typing import Tuple, Dict, Union, List
from abc import ABCMeta, abstractmethod
from pathlib import Path
import time
import argparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader


from pytorch_lightning.lite import LightningLite
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from log import Logger
from tokenization import BertTokenizer
from dataset import CscMlmDataset, collate_csc_mlm_fn_padding
from mask import PinyinConfusionSet, StrokeConfusionSet, Mask
from model import BertForCSC
from load_cbert_weight import load_tf_cbert

class TrainLite(LightningLite, metaclass=ABCMeta):
    @abstractmethod
    def run(self, args):
        self.seed_everything(args.seed)
        self.hparams = args

        
        # 加载数据
        train_loader, test_loader = self.train_dataloader(), self.test_dataloader()
        self.hparams.step_samples = self.hparams.train_batch_size * max(1, self.hparams.gpus) * self.hparams.nodes * \
            self.hparams.accumulation_steps
        self.hparams.total_steps = len(train_loader.dataset) * self.hparams.epochs / self.hparams.step_samples

        optimizer, self.scheduler = self.configure_optimizers()
        self.model, self.optimizer = self.setup(self.model, optimizer)

        train_loader, test_loader = self.setup_dataloaders(train_loader, test_loader)

        # 设置变量
        self.current_step = 1
        self.start_time = time.time()

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch: Dict[str, Tensor]):
        raise NotImplementedError

    def train_epoch(self, train_loader: DataLoader, test_loader: DataLoader):
        for i, batch in enumerate(train_loader):
            self.model.train()
            loss = self.train_step(batch)
            loss = loss / self.accumulation_steps
            self.backward(loss)

            if self.current_step % self.hparams.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.model.zero_grad()
                self.current_step += 1
            
            if self.current_step % self.hparams.report_steps == 0 and self.is_global_zero:
                self.report_and_reset_stats()
                self.start_time = time.time()
            
            if self.current_step % self.hparams.eval_steps == 0:
                self.eval(test_loader)
            
            if self.current_step % self.hparams.save_checkpoint_steps == 0 and self.is_global_zero:
                self.save_model()

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        self.model.train()
        for epoch in range(1, self.hparams.epochs + 1):
            self.epoch = epoch
            self.train_epoch(train_loader, test_loader)
    
    @torch.no_grad()
    def eval(self, test_loader: DataLoader):
        raise NotImplementedError
    
    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
    
    @abstractmethod
    def train_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self):
        raise NotImplementedError
    
    @abstractmethod
    def report_and_reset_stats(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def save_model(self) -> None:
        raise NotImplementedError

class CSCTrainLite(LightningLite):
    def run(self, args):
        self.seed_everything(args.seed)
        self.hparams = args

        # 加载tokenizer
        tokenizer = BertTokenizer(self.hparams.vocab_path)
        self.tokenizer = tokenizer

        # 设置模型分类标签的数量
        self.hparams.num_labels = len(self.tokenizer.vocab)

        # 加载混淆集
        same_py_file = Path(self.hparams.confusions) / 'same_pinyin.txt'
        simi_py_file =Path(self.hparams.confusions) / 'simi_pinyin.txt'
        stroke_file = Path(self.hparams.confusions) / 'same_stroke.txt'
        pinyin = PinyinConfusionSet(tokenizer, same_py_file)
        jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
        stroke = StrokeConfusionSet(tokenizer, stroke_file)

        # 构建mask策略对象
        self.mask = Mask(pinyin, jinyin, stroke)

        # 构建模型
        config = BertConfig.from_pretrained(self.hparams.config_path)
        self.model = BertForCSC(config, self.hparams.num_labels)

        # 模型加载参数
        if self.hparams.from_tf:
            self.model = load_tf_cbert(self.model, Path(self.hparams.pretrained_model_path))
        elif self.hparams.pretrained_model_path is not None:
            self.model.load_state_dict(self.load(self.hparams.pretrained_model_path), strict=False)
        else:
            for n, p in list(self.model.named_parameters()):
                    if "gamma" not in n and "beta" not in n:
                        p.data.normal_(0, 0.02)
        
        # 加载数据
        self.train_loader, self.test_loader = self.train_dataloader(), self.test_dataloader()
        # loader set_dataloader前后长度会发生变化，set后的长度为多设备的平均

        # setup
        self.model, 


    def train_dataloader(self):
        data = CscMlmDataset(self.hparams.seq_length, self.tokenizer, self.mask, self.hparams.train_data_path)
        return DataLoader(data, batch_size=self.hparams.train_batch_size, collate_fn=collate_csc_mlm_fn_padding)
    
    def test_dataloader(self):
        data = CscMlmDataset(self.hparams.seq_length, self.tokenizer, self.mask, self.hparams.test_data_path, mode='test')
        return DataLoader(data, batch_size=self.hparams.test_batch_size, collate_fn=collate_csc_mlm_fn_padding)
    
    def configure_optimizers(self) -> Tuple:
        model = self.model
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                if 'bert' not in n], 'lr': self.hparams.task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, correct_bias=not(self.hparams.bertadam))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_proportion * self.hparams.total_steps),
            num_training_steps=self.hparams.total_steps,
        )
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        self.total_steps = self.compute_warmup()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup * self.total_steps,
            num_training_steps=self.total_steps,
        )
        return optimizer, scheduler
    
    def compute_warmup(self) -> int:
        dataset_size = len(self.train_loader)
        num_devices = max(1, self.hparams.num_devices) * self.hparams.num_nodes

        effective_batch_size = self.trainer.grad_accumulation_steps * num_devices
        num_training_steps = (dataset_size // effective_batch_size) * self.trainer.epochs_num

        return num_training_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Chinese Spell Correct')

    # dataset options
    parser.add_argument("--train_data_path", type=str,
                        help='path of the train dataset')
    parser.add_argument("--test_data_path", type=str,
                        help='path of the test dataset')
    
    # bert config options
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path of the config file.")

    # tokenizer options
    parser.add_argument("--vocab_path", default=None, type=str,
                        help="Path of the vocabulary file.")
    
    # model options
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default=None, type=str,
                        help="Path of the output model.")
    parser.add_argument("--from_tf", action="store_true", default=False,
                        help="whether to load params from tensorflow.")

    # confusions options
    parser.add_argument("--confusions", default=None, type=str,
                        help="Path of confusions data")

    # train options
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Batch size of train dataset.")
    parser.add_argument("--test_batch_size", type=int, default=32,
                        help="Batch size of test dataset.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    
    # optimization options
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warmup ratio value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--optimizer", choices=["adamw", "adafactor"],
                        default="adamw",
                        help="Optimizer type.")
    parser.add_argument("--scheduler", choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                                "constant", "constant_with_warmup"],
                        default="linear", help="Scheduler type.")

    # multi node device
    parser.add_argument('--num_devices', type=int, default=1,
                        help='the number of devices of one node for training')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='the number of node for training')
    parser.add_argument('--strategy', type=str, choices=["dp", "ddp", "ddp_spawn", "deepspeed", "ddp_sharded"], default='ddp',
                        help='Strategy for how to run across multiple devices')

    # metric options
    parser.add_argument('ignore_index', type=int, default=-100,
                        help='metric ignore index')

    args = parser.parse_args()
    return args