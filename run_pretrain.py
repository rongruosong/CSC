# coding=utf-8
import argparse

from pl_model import CSCDataModule, CSCPretrainTransformer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from train_argparse import parse_args


def train_model():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    # 加载数据集
    csc_dm = CSCDataModule(args)

    # 初始化模型
    model = CSCPretrainTransformer(args=args, num_labels=csc_dm.num_labels)

    # 构建Trainer
    modelCheckpoint = ModelCheckpoint(
        dirpath=args.output_model_path,
        save_weights_only=True,
        monitor='val_loss',
        every_n_train_steps=args.val_check_interval,
        filename='{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=20,
        mode='max'
    )
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.num_devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision= 16 if args.fp16 else 32,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.grad_accumulation_steps,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=1,
        callbacks=[LearningRateMonitor("step"), modelCheckpoint],
        deterministic=False
    )
    trainer.fit(model, csc_dm)

