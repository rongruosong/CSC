# coding=utf-8
from tokenization import BertTokenizer
from model import BertForCSC
from mask import PinyinConfusionSet, StrokeConfusionSet, Mask
from pathlib import Path
from addict import Dict
from dataset import CscMlmDataset, collate_csc_mlm_fn_padding
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

    dataset = CscMlmDataset(conf, mask, Path('./data/test_sample.txt'))
    # src = dataset['src'].tolist()
    # tgt = dataset['tgt'].tolist()

    # print([(i, c) for i, c in enumerate(tokenizer.convert_ids_to_tokens(src))])
    # print([(i, c) for i, c in enumerate(tokenizer.convert_ids_to_tokens(tgt))])
    loader = DataLoader(dataset, batch_size=4, num_workers=1, persistent_workers=True, collate_fn=collate_csc_mlm_fn_padding)
    for batch in loader:
        print(batch)
        break
