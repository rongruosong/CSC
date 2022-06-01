# coding=utf-8
from functools import partial
from tokenization import BertTokenizer
from model import BertForCSC
from mask import PinyinConfusionSet, StrokeConfusionSet, Mask
from pathlib import Path
from addict import Dict
from dataset import CscMlmDataset, CscTaskDataset, collate_csc_fn_padding
from torch.utils.data import DataLoader


if __name__ == '__main__':
    tokenizer = BertTokenizer('./cbert/vocab.txt')
    same_py_file = Path('./confusions/same_pinyin.txt')
    simi_py_file = Path('./confusions/simi_pinyin.txt')
    stroke_file = Path('./confusions/same_stroke.txt')
    pinyin = PinyinConfusionSet(tokenizer, same_py_file)
    jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
    print('pinyin conf size:', len(pinyin.confusion))
    print('jinyin conf size:', len(jinyin.confusion))
    stroke = StrokeConfusionSet(tokenizer, stroke_file)

    conf = Dict({'tokenizer': tokenizer, 'seq_length': 36})

    mask = Mask(pinyin, jinyin, stroke)

    # dataset = CscMlmDataset(36, tokenizer, mask, Path('./data/test_sample.txt'))
    # src = dataset['src'].tolist()
    # tgt = dataset['tgt'].tolist()

    # print([(i, c) for i, c in enumerate(tokenizer.convert_ids_to_tokens(src))])
    # print([(i, c) for i, c in enumerate(tokenizer.convert_ids_to_tokens(tgt))])
    dataset = CscTaskDataset(seq_length=46, tokenizer=tokenizer, file_path=Path('./data/sighan_2015/train.tsv'))
    # collate_csc_task_fn_padding = partial(collate_csc_task_fn_padding, mode='train')
    loader = DataLoader(dataset, batch_size=1, num_workers=1, persistent_workers=True, collate_fn=collate_csc_fn_padding)
    for i, batch in enumerate(loader):
        print(i)
