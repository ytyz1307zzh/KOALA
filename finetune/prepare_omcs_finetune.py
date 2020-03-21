'''
 @Date  : 03/14/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
 Prepare the input for ConceptNet language model fine-tuning.
'''

import json
import argparse
import os
from typing import List, Dict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-omcs', type=str, default='../ConceptNet/OMCS/omcs-sentences.txt', help='directory to the original data')
parser.add_argument('-cpnet', type=str, default='../ConceptNet/result/retrieval.json', help='file containing ConceptNet data')
parser.add_argument('-output_dir', type=str, default='../finetune_data', help='output directory')
opt = parser.parse_args()


def read_omcs(omcs_path: str) -> List[str]:
    assert os.path.isfile(omcs_path)
    omcs_file = open(omcs_path, 'r', encoding='utf8')
    omcs_data = omcs_file.readlines()
    print(f'[INFO] {len(omcs_data)} OMCS sentences loaded.')
    omcs_data = omcs_data[1:]  # remove headers
    en_sents = []

    for instance in tqdm(omcs_data, desc='Extract English Sentences'):
        fields = instance.strip().split('\t')
        if len(fields) != 7:
            continue
        lang = fields[4]
        if lang == 'en':
            en_sents.append(fields[1])

    print(f'[INFO] {len(en_sents)} English sentences extracted. ({len(en_sents)/len(omcs_data)*100:.2f}%)')
    return en_sents


def read_cpnet(cpnet_path: str) -> List[str]:
    assert os.path.isfile(cpnet_path)
    cpnet_data = json.load(open(cpnet_path, 'r', encoding='utf8'))
    cpnet_sents = []

    for instance in tqdm(cpnet_data, desc='Load ConceptNet Sentences'):
        cpnet_triples = instance['cpnet']
        for triple in cpnet_triples:
            fields = triple.strip().split(', ')
            assert len(fields) == 11 or len(fields) == 13
            cpnet_sents.append(fields[10])

    return cpnet_sents


def main():
    omcs_sents = read_omcs(opt.omcs)
    cpnet_sents = read_cpnet(opt.cpnet)

    train_path = os.path.join(opt.output_dir, 'omcs_train.txt')
    eval_path = os.path.join(opt.output_dir, 'omcs_eval.txt')

    with open(train_path, 'w', encoding='utf8') as train_file:
        for sent in omcs_sents:
            train_file.write(sent + '\n')
        for sent in cpnet_sents:
            train_file.write(sent + '\n')

    with open(eval_path, 'w', encoding='utf8') as eval_file:
        for sent in cpnet_sents:
            eval_file.write(sent + '\n')


if __name__ == '__main__':
    main()

