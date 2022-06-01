# coding=utf-8
from typing import Dict
from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
from pathlib import Path
from mask import Mask
from torch.utils.data import DataLoader, Dataset
from tokenization import BertTokenizer

class CscDatset(Dataset, metaclass=ABCMeta):
    def __init__(self, file_path: str) -> None:
        self.file = Path(file_path)
        self.sample_size = 0
        self.sample_index = list()
        self._parse_dataset()

    def _parse_dataset(self) -> None:
        with self.file.open(mode='r', encoding='utf-8') as fin:
          self.sample_index.append(0)
          while True:
              line = fin.readline()
              if not line:
                  self.sample_index.pop()
                  break
              self.sample_index.append(fin.tell())
              self.sample_size += 1

    @abstractmethod
    def _create_sample(self, line):
        raise NotImplementedError
    
    def __len__(self) -> int:
        return self.sample_size
    
    def __getitem__(self, index) -> Dict[str, Tensor]:
        if index >= self.sample_size:
            raise IndexError
        index = self.sample_index[index]
        fin = self.file.open(encoding='utf-8')
        fin.seek(index)
        return self._create_sample(fin.readline())

class CscMlmDataset(CscDatset):
    def __init__(self, seq_length: int, tokenizer: BertTokenizer, mask: Mask, file_path: Path):
        super().__init__(file_path=file_path)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask = mask
    
    def _create_sample(self, line):
        tokens = []
        tokens.append(self.tokenizer.cls_token)
        tokens.extend(self.tokenizer.tokenize(line))

        # 根据支持的句子最大长度进行处理
        if len(tokens) > self.seq_length - 1:
            tokens = tokens[:self.seq_length - 1]
        tokens.append(self.tokenizer.sep_token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        src_tokens, tgt_tokens = self.mask(indexed_tokens)
        src_tokens = torch.LongTensor(src_tokens)
        tgt_tokens = torch.LongTensor(tgt_tokens)

        return dict(src=src_tokens, tgt=tgt_tokens)

class CscTaskDataset(CscDatset):
    def __init__(self, 
        seq_length:int, 
        tokenizer: BertTokenizer, 
        file_path: Path, 
        ignore_index: int=-100,
        mode: str='train'
    ) -> None:
        super().__init__(file_path=file_path)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.ignore_index = ignore_index
        self.mode = mode
    
    def _create_sample(self, line: str) -> Dict[str, Tensor]:
        lines = line.strip().split('\t')
        src_tokens = self.tokenizer.tokenize(lines[0])
        if self.mode != 'infer':
            tgt_tokens = self.tokenizer.tokenize(lines[1])
            if len(tgt_tokens) != len(src_tokens):
                print(lines)
            assert len(tgt_tokens) == len(src_tokens), 'the length of src line is not equal to tgt line\'s'

        src_tokens = [self.tokenizer.cls_token] + src_tokens
        if len(src_tokens) > self.seq_length - 1:
            src_tokens = src_tokens[:self.seq_length - 1]
        src_tokens.append(self.tokenizer.sep_token)
        src_tokens = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(src_tokens))

        if self.mode != 'infer':
            tgt_tokens = [self.tokenizer.cls_token] + tgt_tokens
            if len(tgt_tokens) > self.seq_length - 1:
                tgt_tokens = tgt_tokens[:self.seq_length - 1]
            tgt_tokens.append(self.tokenizer.sep_token)
            tgt_tokens = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(tgt_tokens))
            tgt_tokens[0] = self.ignore_index
            tgt_tokens[-1] = self.ignore_index
            return dict(src=src_tokens, tgt=tgt_tokens)
            
        return dict(src=src_tokens)

def collate_csc_fn_padding(batch: Dict[str, Tensor], ignore_index: int=-100, mode: str='train'):
    """
    用于pretrain与task的数据处理
    """
    src_tokens = [item['src'] for item in batch]
    if mode != 'infer':
        tgt_tokens = [item['tgt'] for item in batch]

    token_lens = [token.size(-1) for token in src_tokens]
    max_token_len = max(token_lens)
    
    tokens_id = []
    attention_mask = []
    tgt_ids = []

    for i in range(len(src_tokens)):
        src_token, token_len = src_tokens[i], token_lens[i]
        pad_len = max_token_len - token_len
        src_pad = torch.zeros(pad_len, dtype=torch.long)

        tokens_id.append(torch.cat((src_token, src_pad)))

        mask = torch.ones(token_len, dtype=torch.long)
        attention_mask.append(torch.cat((mask, src_pad)))

        if mode != 'infer':
            tgt_token = tgt_tokens[i]
            tgt_pad = torch.LongTensor([ignore_index] * pad_len)
            tgt_ids.append(torch.cat((tgt_token, tgt_pad)))
    
    input_ids = torch.stack(tokens_id, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    if mode != 'infer':
        tgt_ids = torch.stack(tgt_ids, dim=0)
    else:
        tgt_ids = None

    return input_ids, attention_mask, tgt_ids