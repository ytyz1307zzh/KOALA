'''
 @Date  : 05/07/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import argparse
import json
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import List, Dict
from collections import OrderedDict
import os
BERT_HIDDEN_SIZE = 768

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, required=True, help='fine-tuned model path')
parser.add_argument('-output', type=str, default='wiki_case.json', help='output file path')
parser.add_argument('-text', type=str, default='../finetune_data/eval.txt', help='the eval text during finetuning')
parser.add_argument('-pos', type=str, default='../finetune_data/eval.pos', help='the eval pos during finetuning')
parser.add_argument('-topk', type=int, default=10, help='number of words to keep')
parser.add_argument('-no_cuda', action='store_true', default=False)
opt = parser.parse_args()
opt.cuda = not opt.no_cuda

# assert os.path.isdir(opt.model)
pretrain_model = BertModel.from_pretrained('bert-base-uncased')
finetune_model = BertModel.from_pretrained(opt.model)
if opt.cuda:
    pretrain_model.cuda()
    finetune_model.cuda()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def cos_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    assert vec1.size() == vec2.size() == (BERT_HIDDEN_SIZE,)
    return F.cosine_similarity(vec1, vec2, dim=0)


def find_topk_words(text: str, pos: List[int]):
    input_ids = torch.tensor([tokenizer.encode(text)])
    pos = list(map(lambda x: int(x)+1, pos))  # add <CLS>
    if opt.cuda:
        input_ids = input_ids.cuda()

    pretrain_all_embed = pretrain_model(input_ids)[0].squeeze()
    finetune_all_embed = finetune_model(input_ids)[0].squeeze()

    pretrain_nv_embed = pretrain_all_embed[pos]
    finetune_nv_embed = finetune_all_embed[pos]  # (num_token, embed_size)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    nv_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][pos])

    assert pretrain_nv_embed.size(0) == finetune_nv_embed.size(0) == len(nv_tokens)
    result = []
    for tid in range(len(nv_tokens)):
        word = nv_tokens[tid]

        # on pre-trained BERT
        embed = pretrain_nv_embed[tid]
        word_dict = {}
        for oid in range(input_ids.size(-1)):
            other_word = all_tokens[oid]
            if other_word == word or other_word in word_dict.keys():
                continue
            other_embed = pretrain_all_embed[oid]
            sim = cos_similarity(embed, other_embed)
            word_dict[other_word] = sim.item()
        word_dict = OrderedDict(sorted(word_dict.items(), key=lambda d: d[1], reverse=True))  # sort by similarity
        pretrain_topk = list(word_dict.items())[:opt.topk]

        # on fine-tuned BERT
        embed = finetune_nv_embed[tid]
        word_dict = {}
        for oid in range(input_ids.size(-1)):
            other_word = all_tokens[oid]
            if other_word == word or other_word in word_dict.keys():
                continue
            other_embed = finetune_all_embed[oid]
            sim = cos_similarity(embed, other_embed)
            word_dict[other_word] = sim.item()
        word_dict = OrderedDict(sorted(word_dict.items(), key=lambda d: d[1], reverse=True))  # sort by similarity
        finetune_topk = list(word_dict.items())[:opt.topk]
        result.append({'token': word,
                       'pretrain': {word: score for word, score in pretrain_topk},
                       'finetune': {word: score for word, score in finetune_topk}})

    return result


def main():
    with open(opt.text, encoding="utf-8") as f:
        textlines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        text_list = textlines

    with open(opt.pos, encoding="utf-8") as f:
        poslines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        pos_list = list(map(lambda s: s.strip().split(), poslines))

    assert len(text_list) == len(pos_list)
    result = []

    for i in tqdm(range(len(text_list))):
        scores = find_topk_words(text_list[i], pos_list[i])
        result.append({'id': i,
                       'paragraph': text_list[i],
                       'result': scores})

    json.dump(result, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()


