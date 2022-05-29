# coding=utf-8
"""

"""
from typing import Union, Tuple
from pathlib import Path
from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import BertConfig, get_linear_schedule_with_warmup

from log import Logger
from tokenization import BertTokenizer
from dataset import CscMlmDataset, collate_csc_fn_padding
from mask import PinyinConfusionSet, StrokeConfusionSet, Mask
from model import BertForCSC
from load_cbert_weight import load_tf_cbert
from torchmetrics import MeanMetric, Accuracy
from metrics import CSCScore

class CSCTransformer(LightningModule, metaclass=ABCMeta):
    def __init__(self, args, num_labels):
        super().__init__()
        self.args = args
        
        # 构建模型
        config = BertConfig.from_pretrained(self.args.config_path)
        self.model = BertForCSC(config, num_labels, self.args.ignore_index)

        # 模型加载参数
        if self.args.from_tf:
            self.model = load_tf_cbert(self.model, Path(self.args.pretrained_model_path))
        elif self.args.pretrained_model_path is not None:
            self.model.load_state_dict(torch.load(self.args.pretrained_model_path), strict=False)
        else:
            for n, p in list(self.model.named_parameters()):
                    if "gamma" not in n and "beta" not in n:
                        p.data.normal_(0, 0.02)
    
    def configure_optimizers(self) -> Tuple:
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        self.total_steps, self.num_warmup_steps = self.compute_warmup(self.args.warmup)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.devices) * self.trainer.num_nodes

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        """
        返回训练的最大step, warmup的step
        """
        if num_training_steps < 0:
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

class CSCPretrainTransformer(CSCTransformer):
    def __init__(self, args, num_labels):
        super().__init__(args, num_labels)

        self.cur_step = 0

        self.report_loss = MeanMetric()
        self.report_acc = Accuracy(ignore_index=self.args.ignore_index)

        self.val_loss = MeanMetric()
        self.val_acc = Accuracy(ignore_index=self.args.ignore_index)
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx) -> Tensor:
        input_ids, attention_mask, tgt_ids = batch
        outputs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=tgt_ids)
        loss, logits = outputs[0:2]
        preds = torch.argmax(logits, axis=1)
        self.report_acc(preds, tgt_ids)
        self.log('batch_loss', loss, prog_bar=True, rank_zero_only=True) # 只输出该batch的loss
        self.log('batch_acc', self.report_acc, prog_bar=True, rank_zero_only=True) # 当前batch的accuracy
        return loss
    
    def training_step_end(self, outs):
        # 这里cur_step是实际optim step的accumulate_grad_batches倍
        self.cur_step += 1
        self.report_loss.update(outs)

        # 这里考虑accumulate_grad_batches次training_step 算一次，
        if (self.cur_step % (self.trainer.accumulate_grad_batches * self.args.report_steps)) == 0:
            self.log('report_loss', self.report_loss.compute(), rank_zero_only=True)
            self.report_loss.reset()
            self.log('report_acc', self.report_acc.compute(), rank_zero_only=True) # 全部batches的accuracy
            self.report_acc.reset()
        return outs

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, tgt_ids = batch
        outputs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=tgt_ids)
        loss, logits = outputs[0:2]

        self.val_loss(loss)
        self.val_acc(logits, tgt_ids)
        self.log('val_loss', loss)
        return loss
    
    def validation_epoch_end(self, outputs):
        self.log('val_epoch_loss', self.val_loss.compute())
        self.val_loss.reset()

        self.log('val_epoch_acc', self.val_acc.compute())
        self.val_acc.reset()

class CSCTaskTransformer(CSCTransformer):
    def __init__(self, args, num_labels):
        super().__init__(args, num_labels)

        self.cur_step = 0

        self.report_loss = MeanMetric()
        self.report_metric = CSCScore(ignore_index=self.args.ignore_index)

        self.val_loss = MeanMetric()
        self.val_metric = CSCScore(ignore_index=self.args.ignore_index)
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx) -> Tensor:
        input_ids, attention_mask, tgt_ids = batch
        outputs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=tgt_ids)
        loss, logits = outputs[0:2]
        preds = torch.argmax(logits, axis=1)
        metric = self.report_metric(input_ids, logits, tgt_ids)
        batch_metric = {}
        for k, v in metric.items():
            batch_metric['batch_' + k] = v 

        self.log('batch_loss', loss, prog_bar=True, rank_zero_only=True) # 只输出该batch的loss
        self.log_dict(batch_metric, prog_bar=True, rank_zero_only=True) # 当前batch的metric
        return loss
    
    def training_step_end(self, outs):
        # 这里cur_step是实际optim step的accumulate_grad_batches倍
        self.cur_step += 1
        self.report_loss.update(outs)

        # 这里考虑accumulate_grad_batches次training_step 算一次，
        if (self.cur_step % (self.trainer.accumulate_grad_batches * self.args.report_steps)) == 0:
            self.log('report_loss', self.report_loss.compute(), rank_zero_only=True)
            self.report_loss.reset()

            metric = self.report_acc.compute()
            report_metric = {'report_' + k : v for k, v in metric.items()}
            self.log_dict(report_metric, rank_zero_only=True) # 全部batches的accuracy
            self.report_acc.reset()
        return outs

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, tgt_ids = batch
        outputs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=tgt_ids)
        loss, logits = outputs[0:2]

        self.val_loss(loss)
        self.val_metric(input_ids, logits, tgt_ids)
        self.log('val_loss', loss)
        return loss
    
    def validation_epoch_end(self, outputs):
        self.log('val_epoch_loss', self.val_loss.compute())
        self.val_loss.reset()
        
        metric = self.val_acc.compute()
        val_metric = {'val_epoch_' + k : v for k, v in metric.items()}
        self.log_dict(val_metric)
        self.val_acc.reset()

class CSCPretrainTransformer(LightningModule):
    def __init__(self, args, num_labels):
        super().__init__()
        self.args = args
        
        # 构建模型
        config = BertConfig.from_pretrained(self.args.config_path)
        self.model = BertForCSC(config, num_labels, self.args.ignore_index)

        # 模型加载参数
        if self.args.from_tf:
            self.model = load_tf_cbert(self.model, Path(self.args.pretrained_model_path))
        elif self.args.pretrained_model_path is not None:
            self.model.load_state_dict(torch.load(self.args.pretrained_model_path), strict=False)
        else:
            for n, p in list(self.model.named_parameters()):
                    if "gamma" not in n and "beta" not in n:
                        p.data.normal_(0, 0.02)
        
        self.cur_step = 0
        # self.report_loss = torch.tensor(0.0, device=self.device)
        # self.report_loss = []
        self.report_loss = MeanMetric()
        self.report_acc = Accuracy(ignore_index=self.args.ignore_index)

        self.val_loss = MeanMetric()
        self.val_acc = Accuracy(ignore_index=self.args.ignore_index)
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, tgt_ids = batch
        outputs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=tgt_ids)
        loss, logits = outputs[0:2]
        preds = torch.argmax(logits, axis=1)
        self.report_acc(preds, tgt_ids)
        self.log('batch_loss', loss, prog_bar=True) # 只输出该batch的loss
        self.log('batch_acc', self.report_acc, prog_bar=True) # 当前batch的accuracy
        return loss
    
    def training_step_end(self, outs):
        # 这里cur_step是实际optim step的accumulate_grad_batches倍
        self.cur_step += 1
        # self.report_loss += outs.detach()
        # self.report_loss.append(outs.detach())
        self.report_loss.update(outs)

        # 这里考虑accumulate_grad_batches次training_step 算一次，
        if (self.cur_step % (self.trainer.accumulate_grad_batches * self.args.report_steps)) == 0:
            # self.log('report_loss', self.report_loss / self.trainer.accumulate_grad_batches * self.args.report_steps)
            # self.report_loss = torch.tensor(0.0, device=self.device)
            # self.log('report_loss', sum(self.report_loss) / len(self.report_loss))
            # self.report_loss = []
            self.log('report_loss', self.report_loss.compute())
            self.report_loss.reset()
            self.log('report_acc', self.report_acc.compute()) # 全部batches的accuracy
            self.report_acc.reset()
        """
        # 也可以采用这种方式，report_steps 输出一次
        if self.cur_step % self.args.report_steps == 0:
            self.log('report_loss', self.report_loss / self.args.report_steps)
            self.report_loss = torch.tensor(0.0, device=self.device)
            # self.log('report_loss', self.report_loss.compute())
            # self.report_loss.reset()
            self.log('report_acc', self.report_acc.compute()) # 全部batches的accuracy
            self.report_acc.reset()
        """
        return outs
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, tgt_ids = batch
        outputs = self(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=tgt_ids)
        loss, logits = outputs[0:2]

        self.val_loss(loss)
        self.val_acc(logits, tgt_ids)
        self.log('val_loss', loss)
        return loss
    
    def validation_epoch_end(self, outputs):
        self.log('val_epoch_loss', self.val_loss.compute())
        self.val_loss.reset()

        self.log('val_epoch_acc', self.val_acc.compute())
        self.val_acc.reset()

    def configure_optimizers(self) -> Tuple:
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        self.total_steps, self.num_warmup_steps = self.compute_warmup(self.args.warmup)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.devices) * self.trainer.num_nodes

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def compute_warmup(self, num_warmup_steps: Union[int, float]) -> Tuple[int, int]:
        """
        返回训练的最大step, warmup的step
        """
        if num_training_steps < 0:
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

class CSCDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        self.hparams = args
        # 加载tokenizer
        tokenizer = BertTokenizer(self.hparams.vocab_path)
        self.tokenizer = tokenizer

        # 设置模型分类标签的数量
        self.num_labels = len(self.tokenizer.vocab)

        # 加载混淆集
        same_py_file = Path(self.hparams.confusions) / 'same_pinyin.txt'
        simi_py_file =Path(self.hparams.confusions) / 'simi_pinyin.txt'
        stroke_file = Path(self.hparams.confusions) / 'same_stroke.txt'
        pinyin = PinyinConfusionSet(tokenizer, same_py_file)
        jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
        stroke = StrokeConfusionSet(tokenizer, stroke_file)

        # 构建mask策略对象
        self.mask = Mask(pinyin, jinyin, stroke)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_data = CscMlmDataset(self.hparams.seq_length, self.tokenizer, self.mask, self.hparams.train_data_path)
            self.val_data = CscMlmDataset(self.hparams.seq_length, self.tokenizer, self.mask, self.hparams.test_data_path, mode='test')

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_data = CscMlmDataset(self.hparams.seq_length, self.tokenizer, self.mask, self.hparams.test_data_path, mode='test')
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.train_batch_size, collate_fn=collate_csc_fn_padding)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.hparams.test_batch_size, collate_fn=collate_csc_fn_padding)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.test_batch_size, collate_fn=collate_csc_fn_padding)

