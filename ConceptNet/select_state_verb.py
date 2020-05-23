'''
 @Date  : 02/21/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import json
import os
import argparse
from tqdm import tqdm
from typing import List, Dict, Set
from Stemmer import PorterStemmer
from collections import OrderedDict
stemmer = PorterStemmer()

parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', default='../data', type=str, help='directory to the dataset')
parser.add_argument('-output', default='./result/state_verb_5.json', help='output path')
parser.add_argument('-keep_thres', default=5, type=int, help='the threshold of appearance frequency to keep the verb')
opt = parser.parse_args()

create_verb = {}
move_verb = {}
destroy_verb = {}
STOP_VERBS = {'is', 'was', 'are', 'were', 'am', 'be', 'being', 'been', '\'s', 'have', 'has'}


def count_stat(verb_dict: Dict[str, int]):
    """
    Calculate the statistics of the verbs.
    """
    total_verbs = len(verb_dict)
    print(f'Total number of verbs: {total_verbs}')
    discard_list = []

    freq_1, freq_3, freq_5 = 0, 0, 0
    for key, val in verb_dict.items():
        if val <= 1:
            freq_1 += 1
        if val <= 3:
            freq_3 += 1
        if val <= 5:
            freq_5 += 1
        if val < opt.keep_thres:
            discard_list.append(key)
    print(f'Verbs with frequency = 1: {freq_1} ({(freq_1 / total_verbs * 100):.2f}%)')
    print(f'Verbs with frequency <= 3: {freq_3} ({(freq_3 / total_verbs * 100):.2f}%)')
    print(f'Verbs with frequency <= 5: {freq_5} ({(freq_5 / total_verbs * 100):.2f}%)')

    for key in discard_list:
        verb_dict.pop(key)
    print(f'Current threshold: {opt.keep_thres}, verbs with frequency less than this threshold will be discarded.')
    print(f'Retain ratio: {len(verb_dict) / total_verbs * 100:.2f}%')


def update_dict(verb_dict: Dict[str, int], verbs: List[str]):
    """
    Update the frequency of verbs.
    """
    for verb in verbs:
        if verb in STOP_VERBS:
            continue
        verb = stemmer.stem(verb)
        if verb in verb_dict:
            verb_dict[verb] += 1
        else:
            verb_dict[verb] = 1


def select_verb_from_dataset(dataset: List[Dict]):
    """
    Collect high-frequency co-appearing verbs from the dataset
    """
    for instance in dataset:
        gold_state_seq = instance['gold_state_seq']
        sentence_list = instance['sentence_list']
        paragraph = instance['paragraph']
        tokens = paragraph.strip().split()
        verb_mentions = [sentence['verb_mention'] for sentence in sentence_list]
        assert len(gold_state_seq) == len(verb_mentions)

        for sent_idx in range(len(gold_state_seq)):
            state = gold_state_seq[sent_idx]

            if state in ['C', 'M', 'D']:
                # TODO: potential constraint: the entity should appear in this sentence?
                verb_idxs = verb_mentions[sent_idx]
                verbs = [tokens[idx] for idx in verb_idxs]

                if state == 'C':
                    update_dict(create_verb, verbs)
                elif state == 'M':
                    update_dict(move_verb, verbs)
                elif state == 'D':
                    update_dict(destroy_verb, verbs)


def main():
    train_set = json.load(open(os.path.join(opt.data_dir, 'train.json'), 'r', encoding='utf-8'))

    select_verb_from_dataset(train_set)

    print('CREATE')
    count_stat(create_verb)
    print('MOVE')
    count_stat(move_verb)
    print('DESTROY')
    count_stat(destroy_verb)

    result = {'create': sorted(create_verb.keys()),
              'move': sorted(move_verb.keys()),
              'destroy': sorted(destroy_verb.keys())
              }

    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)



if __name__ == "__main__":
    main()
