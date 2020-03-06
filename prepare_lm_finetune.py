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
parser.add_argument('-wiki', type=str, default='wiki/wiki_para_50.json', help='file containing wiki data')
parser.add_argument('-output_dir', type=str, default='finetune_data', help='output directory')
opt = parser.parse_args()


def main():
    wiki_data = json.load(open(opt.wiki, 'r', encoding='utf-8'))
    print('[INFO] Loaded {} instances from wiki file {}'.format(len(wiki_data), opt.wiki))
    train_text = []
    eval_text = []

    for instance in wiki_data:

        ori_para = instance['paragraph']
        wiki_paras = instance['wiki']
        train_text.append(ori_para)
        eval_text.append(ori_para)

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
