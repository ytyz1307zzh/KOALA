'''
 @Date  : 03/06/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
 Prepare the input for language model fine-tuning.
'''

import json
import argparse
import os
from typing import List, Dict

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', type=str, default='../data', help='directory to the original data')
parser.add_argument('-wiki', type=str, default='../wiki/wiki_para_50.json', help='the retrieved wiki paragraphs')
parser.add_argument('-output_dir', type=str, default='../finetune_data', help='output directory')
opt = parser.parse_args()


def read_dataset(dataset: List[Dict]):
    text_list = []
    para_ids = []

    for instance in dataset:
        id_ = instance['id']
        if id_ not in para_ids:
            text_list.append(instance['paragraph'])
            para_ids.append(id_)

    return text_list, para_ids


def main():

    test_set = json.load(open(os.path.join(opt.data_dir, 'test.json'), 'r', encoding='utf-8'))
    wiki_data = json.load(open(opt.wiki, 'r', encoding='utf-8'))
    print('[INFO] Loaded {} instances from wiki file {}'.format(len(wiki_data), opt.wiki))
    train_text = []
    eval_text = []

    test_paras, test_ids = read_dataset(test_set)

    eval_text.extend(test_paras)

    for instance in wiki_data:

        para_id = instance['para_id']
        wiki_paras = instance['wiki']

        for wiki in wiki_paras:
            wiki_text = wiki['text']
            train_text.append(wiki_text)

    print(f'[INFO] Training instances: {len(train_text)}')
    print(f'[INFO] Eval instances: {len(eval_text)}')

    with open(os.path.join(opt.output_dir, 'train.txt'), 'w', encoding='utf-8') as train_outfile:
        for para in train_text:
            train_outfile.write(para + '\n')

    with open(os.path.join(opt.output_dir, 'eval.txt'), 'w', encoding='utf-8') as eval_outfile:
        for para in eval_text:
            eval_outfile.write(para + '\n')

    print('Finished.')


if __name__ == "__main__":
    main()
