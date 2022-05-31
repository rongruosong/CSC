# coding=utf-8
"""
copy from https://github.com/liushulinle/PLOME/blob/main/pre_train_src/mask.py
"""
from typing import List, Tuple
import random
import copy
from pathlib import Path
from tokenization import BertTokenizer

class ConfusionSet:
    def __init__(self, tokenizer: BertTokenizer, in_file: Path):
        self.tokenizer = tokenizer
        self.confusion = self._load_confusion(in_file)

    def _str2idstr(self, string: str):
        ids = [self.tokenizer.vocab.get(x, -1) for x in string]
        if min(ids) < 0:
            return None
        ids = ' '.join(map(str, ids))
        return ids
 
    def _load_confusion(self, in_file: Path):
        pass

    def get_confusion_item_by_ids(self, token_id: int):
        confu = self.confusion.get(token_id, None)
        if confu is None:
            return None
        return confu[random.randint(0,len(confu) - 1)]

    def get_confusion_item_by_unicode(self, key_unicode: str):
        if len(key_unicode) == 1:
            keyid = self.tokenizer.vocab.get(key_unicode, None)
        else:
            keyid = self._str2idstr(key_unicode)
        if keyid is None:
            return None
        confu = self.confusion.get(keyid, None)
        if confu is None:
            return None
        return confu[random.randint(0, len(confu) - 1)]

class PinyinConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file: Path):
        confusion_datas = {}
        for line in in_file.open(encoding='utf-8'):
            line = line.strip()     #.decode('utf-8')
            tmps = line.split('\t')
            if len(tmps) != 2:
                continue
            key = tmps[0]
            values = tmps[1].split()
            if len(key) != 1:
                continue
            all_ids = set()
            keyid = self.tokenizer.vocab.get(key, None)
            if keyid is None:
                continue
            for k in values:
                if self.tokenizer.vocab.get(k, None) is not None:
                    all_ids.add(self.tokenizer.vocab[k])
            all_ids = list(all_ids)
            if len(all_ids) > 0:
                confusion_datas[keyid] = all_ids
        return confusion_datas

class StrokeConfusionSet(ConfusionSet):
    def _load_confusion(self, in_file: Path):
        confusion_datas = {}
        for line in in_file.open(encoding='utf-8'):
            line = line.strip()     #.decode('utf-8')
            tmps = line.split(',')
            if len(tmps) < 2:
                continue
            values = tmps
            all_ids = set()
            for k in values:
                if k in self.tokenizer.vocab:
                    all_ids.add(self.tokenizer.vocab[k])
            all_ids = list(all_ids)
            for i, k in enumerate(all_ids):
                confusion_datas[k] = all_ids[0:i] + all_ids[i+1:]
        return confusion_datas

class Mask(object):
    def __init__(self, same_py_confusion, simi_py_confusion, sk_confusion, ignore_index=-100):
        self.same_py_confusion = same_py_confusion
        self.simi_py_confusion = simi_py_confusion
        self.sk_confusion = sk_confusion
        self.ignore_index = ignore_index
        self.config = {'same_py': 0.3, 'simi_py': 0.3, 'stroke': 0.15, 'random': 0.1, 'keep': 0.15, 'global_rate': 0.15}
        self.same_py_thr = self.config['same_py'] 
        self.simi_py_thr = self.config['same_py'] + self.config['simi_py']
        self.stroke_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke']
        self.random_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke'] + self.config['random']
        self.keep_thr = self.config['same_py'] + self.config['simi_py'] + self.config['stroke'] + self.config['random'] + self.config['keep']
        self.invalid_ids = set([self.same_py_confusion.tokenizer.vocab.get('UNK'),
                               self.same_py_confusion.tokenizer.vocab.get('[CLS]'),
                               self.same_py_confusion.tokenizer.vocab.get('[SEP]'),
                               self.same_py_confusion.tokenizer.vocab.get('[UNK]')])

        self.all_token_ids = [int(x) for x in self.same_py_confusion.tokenizer.vocab.values()]
        self.n_all_token_ids = len(self.all_token_ids) - 1
    
    def __call__(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        return self.mask_process(token_ids)

    def get_mask_method(self) -> str:
        prob = random.random()
        if prob <= self.same_py_thr:
            return 'pinyin'
        elif prob <= self.simi_py_thr:
            return 'jinyin'
        elif prob <= self.stroke_thr:
            return 'stroke'
        elif prob <= self.random_thr:
            return 'random'
        elif prob <= self.keep_thr:
            return 'keep'
        return 'pinyin'

    def mask_process(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        valid_ids = [idx for (idx, v) in enumerate(token_ids) if v not in self.invalid_ids]
        src_tokens = copy.deepcopy(token_ids)
        tgt_tokens = [self.ignore_index] * len(token_ids)
        n_masked = int(len(valid_ids) * self.config['global_rate'])
        if n_masked < 1:
            n_masked = 1
        random.shuffle(valid_ids)
        for pos in valid_ids[:n_masked]:
            method = self.get_mask_method()
            if method == 'pinyin':
                new_c = self.same_py_confusion.get_confusion_item_by_ids(token_ids[pos])
                if new_c is not None:
                    src_tokens[pos] = new_c
                    tgt_tokens[pos] = token_ids[pos]
            elif method == 'jinyin':
                new_c = self.simi_py_confusion.get_confusion_item_by_ids(token_ids[pos])
                if new_c is not None:
                    src_tokens[pos] = new_c
                    tgt_tokens[pos] = token_ids[pos]
            elif method == 'stroke':
                new_c = self.sk_confusion.get_confusion_item_by_ids(token_ids[pos]) 
                if new_c is not None:
                    src_tokens[pos] = new_c
                    tgt_tokens[pos] = token_ids[pos]
            elif method == 'random':
                new_c = self.all_token_ids[random.randint(0, self.n_all_token_ids)]
                if new_c is not None:
                    src_tokens[pos] = new_c
                    tgt_tokens[pos] = token_ids[pos]
            elif method == 'keep': 
                tgt_tokens[pos] = token_ids[pos]
        return src_tokens, tgt_tokens
