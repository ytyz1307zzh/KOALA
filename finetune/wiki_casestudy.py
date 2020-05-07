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

    pretrain_embed = pretrain_model(input_ids)[0].squeeze()
    finetune_embed = finetune_model(input_ids)[0].squeeze()

    pretrain_embed = pretrain_embed[pos]
    finetune_embed = finetune_embed[pos]  # (num_token, embed_size)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0][pos])

    assert pretrain_embed.size(0) == finetune_embed.size(0) == len(tokens)
    result = []
    for tid in range(len(tokens)):
        word = tokens[tid]

        # on pre-trained BERT
        embed = pretrain_embed[tid]
        word_list = []
        for oid in range(len(tokens)):
            if tid == oid:
                continue
            other_embed = pretrain_embed[oid]
            other_word = tokens[oid]
            sim = cos_similarity(embed, other_embed)
            word_list.append({'token': other_word,
                              'similarity': sim.item()})
        word_list = sorted(word_list, key=lambda d: d['similarity'], reverse=True)
        pretrain_topk = word_list[:opt.topk]

        # on fine-tuned BERT
        embed = finetune_embed[tid]
        word_list = []
        for oid in range(len(tokens)):
            if tid == oid:
                continue
            other_embed = finetune_embed[oid]
            other_word = tokens[oid]
            sim = cos_similarity(embed, other_embed)
            word_list.append({'token': other_word,
                              'similarity': sim.item()})
        word_list = sorted(word_list, key=lambda d: d['similarity'], reverse=True)
        finetune_topk = word_list[:opt.topk]
        result.append({'token': word,
                       'pretrain': pretrain_topk,
                       'finetune': finetune_topk})

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

    for i in range(len(text_list)):
        scores = find_topk_words(text_list[i], pos_list[i])
        result.append({'id': i,
                       'paragraph': text_list[i],
                       'result': scores})

    json.dump(result, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()


