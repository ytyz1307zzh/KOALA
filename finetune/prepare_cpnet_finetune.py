'''
 @Date  : 03/21/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
 Prepare the input for ConceptNet language model fine-tuning.
'''

import json
import argparse
import os
from typing import List
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('-cpnet', type=str, default='../ConceptNet/result/retrieval.json', help='file containing ConceptNet data')
parser.add_argument('-output_dir', type=str, default='../finetune_data', help='output directory')
opt = parser.parse_args()


def read_cpnet(cpnet_path: str) -> (List[str], List[str]):
    assert os.path.isfile(cpnet_path)
    cpnet_data = json.load(open(cpnet_path, 'r', encoding='utf8'))
    cpnet_sents = []
    mask_pos_list = []

    for instance in tqdm(cpnet_data, desc='Load ConceptNet Sentences'):
        cpnet_triples = instance['cpnet']

        for triple in cpnet_triples:
            fields = triple.strip().split(', ')
            assert len(fields) == 11 or len(fields) == 13
            sentence = fields[10]

            subj = fields[2]
            obj = fields[5]
            subj_len = len(subj.strip().split('_'))
            obj_len = len(obj.strip().split('_'))
            total_len = len(sentence.strip().split())
            assert subj_len > 0 and obj_len > 0 and total_len > 0

            index_list = list(range(0, total_len))
            subj_idx = index_list[:subj_len]
            obj_idx = index_list[-obj_len:]
            rel_idx = index_list[subj_len:(total_len - obj_len)]
            assert subj_idx + rel_idx + obj_idx == index_list

            # deal with subject mask
            if subj_len == 1:
                mask_idx = subj_idx
                cpnet_sents.append(sentence)
                mask_pos_list.append(mask_idx)

            else:  # multiple words
                rand_idx = random.sample(subj_idx, k=subj_len // 2)
                first_mask_idx = sorted(rand_idx)
                cpnet_sents.append(sentence)
                mask_pos_list.append(first_mask_idx)

                second_mask_idx = [idx for idx in subj_idx if idx not in first_mask_idx]
                cpnet_sents.append(sentence)
                mask_pos_list.append(second_mask_idx)

            # deal with object mask
            if obj_len == 1:
                mask_idx = obj_idx
                cpnet_sents.append(sentence)
                mask_pos_list.append(mask_idx)

            else:  # multiple words
                rand_idx = random.sample(obj_idx, k=obj_len // 2)
                first_mask_idx = sorted(rand_idx)
                cpnet_sents.append(sentence)
                mask_pos_list.append(first_mask_idx)

                second_mask_idx = [idx for idx in obj_idx if idx not in first_mask_idx]
                cpnet_sents.append(sentence)
                mask_pos_list.append(second_mask_idx)

            # deal with relation mask
            # mask all relation words since there are limited types of relations
            cpnet_sents.append(sentence)
            mask_pos_list.append(rel_idx)

    return cpnet_sents, mask_pos_list


def main():
    cpnet_sents, mask_pos_list = read_cpnet(opt.cpnet)
    assert len(cpnet_sents) == len(mask_pos_list)

    text_outpath = os.path.join(opt.output_dir, 'cpnet.txt')
    mask_outpath = os.path.join(opt.output_dir, 'cpnet_mask.txt')

    with open(text_outpath, 'w', encoding='utf8') as text_file:
        for sent in cpnet_sents:
            text_file.write(sent + '\n')

    with open(mask_outpath, 'w', encoding='utf8') as mask_file:
        for mask_pos in mask_pos_list:
            mask_pos = list(map(str, mask_pos))
            mask_file.write(' '.join(mask_pos) + '\n')

    print(f'Obtained a total of {len(cpnet_sents)} sentences')


if __name__ == '__main__':
    main()

