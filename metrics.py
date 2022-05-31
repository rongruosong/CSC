# coding=utf-8
from typing import Callable, Optional, Tuple, Dict, Any
import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import DataType
from torchmetrics.utilities.checks import _check_shape_and_type_consistency

def _drop_negative_ignored_indices(
    inputs: Tensor,
    preds: Tensor, 
    target: Tensor, 
    ignore_index: int, 
    mode: DataType
) -> Tuple[Tensor, Tensor, Tensor]:
    """Remove negative ignored indices.
    Return:
        Tensors of preds and target without negative ignore target values.
    """
    if mode == mode.MULTIDIM_MULTICLASS and preds.dtype == torch.float:
        # In case or multi-dimensional multi-class with logits
        n_dims = len(preds.shape)
        num_classes = preds.shape[-1]
        # flatten: [N, ..., C] -> [N', C]
        preds = preds.reshape(-1, num_classes)
        target = target.reshape(-1)
        inputs = inputs.reshape(-1)

    if mode in [mode.MULTICLASS, mode.MULTIDIM_MULTICLASS]:
        inputs = inputs[target != ignore_index]
        preds = preds[target != ignore_index]
        target = target[target != ignore_index]

    return inputs, preds, target

class CSCScore(Metric):
    def __init__(self, 
        ignore_index: Optional[int] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        # self.id2label = id2label
        self.ignore_index = ignore_index
        self.add_state('total_gold', tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('total_pred', tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('right_pred', tensor(0, dtype=torch.float), dist_reduce_fx='sum')
        self.add_state('check_right_pred', tensor(0, dtype=torch.float), dist_reduce_fx='sum')

    def _calculate(self, total_gold: Tensor, total_pred: Tensor, right_pred: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args
        --------
            origin: 原本的标签
            found: 预测出来的标签
            right: 预测标签中正确的标签

        Returns
        ---------
        计算出来的精确率 召回率  f1值
        """
        total_gold, total_pred, right_pred = total_gold.float(), total_pred.float(), right_pred.float()
        
        prec = torch.tensor(0.0, device=self.device) if total_pred == 0 else (right_pred / total_pred)
        rec = torch.tensor(0.0, device=self.device) if total_gold == 0 else (right_pred / total_gold)
        f1 = torch.tensor(0.0, device=self.device) if prec + \
            rec == 0 else (2 * rec * prec) / (prec + rec)

        return rec, prec, f1

    def compute(self) -> Dict[str, Tensor]:
        detection_rec, detection_prec, detection_f1 = self._calculate(self.total_gold, self.total_pred, self.check_right_pred)
        correction_rec, correction_pre, correction_f1 = self._calculate(self.total_gold, self.total_pred, self.right_pred)
        return {
            'det_prec': detection_prec, 
            'det_rec': detection_rec, 
            'det_f1': detection_f1, 
            'cor_prec': correction_pre,
            'cor_rec': correction_rec,
            'cor_f1': correction_f1
        }

    def update(self, inputs: Tensor, preds: Tensor, target: Tensor) -> None:
        '''
        inputs: [N,...]
        preds: [N,...,C]
        target: [N,...]
        '''
        if self.ignore_index is not None and self.ignore_index < 0:
            mode, _ = _check_shape_and_type_consistency(preds.transpose(1, -1), target)
            inputs, preds, target = _drop_negative_ignored_indices(inputs, preds, target, self.ignore_index, mode)
        
        if preds.is_floating_point():
            preds = preds.argmax(dim=-1)
        
        self.total_gold += sum(inputs != target)
        self.total_pred += sum(inputs != preds)
        check_right_pred = (inputs != target) & (inputs != preds)
        self.check_right_pred += sum(check_right_pred)
        self.right_pred += sum(check_right_pred & (target == preds))
