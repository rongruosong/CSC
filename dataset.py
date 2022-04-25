# coding=utf-8
import torch
from pathlib import Path
from mask import Mask
from torch.utils.data import DataLoader, Dataset

class CscMlmDataset(Dataset):
    def __init__(self, args, mask: Mask, file_path: Path, mode: str='train'):
        self.sample_size = 0
        self.sample_index = list()
        self.file = file_path
        self.tokenizer = args.tokenizer
        self.seq_length = args.seq_length
        self.mask = mask
        self.mode = mode
        self._parse_dataset()

    def _parse_dataset(self):
        with self.file.open(mode='r', encoding='utf-8') as fin:
          self.sample_index.append(0)
          while True:
              line = fin.readline()
              if not line:
                  self.sample_index.pop()
                  break
              self.sample_index.append(fin.tell())
              self.sample_size += 1
    
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

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        if index >= self.sample_size:
            raise IndexError
        index = self.sample_index[index]
        fin = self.file.open(encoding='utf-8')
        fin.seek(index)
        return self._create_sample(fin.readline())

def collate_csc_mlm_fn_padding(batch, mode='train'):
    src_tokens = [item['src'] for item in batch]
    tgt_tokens = [item['tgt'] for item in batch]

    token_lens = [token.size(-1) for token in src_tokens]
    max_token_len = max(token_lens)
    
    tokens_id = []
    attention_mask = []
    tgt_ids = []
    for src_token, tgt_token, token_len in zip(src_tokens, tgt_tokens, token_lens):
        pad_len = max_token_len - token_len
        src_pad = torch.zeros(pad_len, dtype=torch.long)
        tgt_pad = torch.LongTensor([-100] * pad_len)

        tokens_id.append(torch.cat((src_token, src_pad)))

        mask = torch.ones(token_len, dtype=torch.long)
        attention_mask.append(torch.cat((mask, src_pad)))

        tgt_ids.append(torch.cat((tgt_token, tgt_pad)))
    
    input_ids = torch.stack(tokens_id, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    tgt_ids = torch.stack(tgt_ids, dim=0)

    return input_ids, attention_mask, tgt_ids
