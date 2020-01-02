'''
 @Date  : 01/02/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
from tqdm import tqdm
import argparse
import json
from typing import Dict, List


def get_pos(concept: str) -> (str, str):
    """
    If the concept has specific pos, extract it
    """
    if concept.endswith('/n') or concept.endswith('/v') or concept.endswith('/a') or concept.endswith('/r'):
        return concept[:-2], concept[-1]
    else:
        return concept, None


def add_freq(concept: str, freq: Dict):
    if concept not in freq:
        freq[concept] = 1
    else:
        freq[concept] += 1


def compute_freq(opt):
    """
    Prune the less frequently appeared concepts from ConceptNet.
    """
    freq = {}

    with open(opt.cpnet, 'r', encoding='utf8') as fin:
        cp_lines = fin.readlines()

    print('Start computing frequency...')
    for line in tqdm(cp_lines):

        data = line.strip().split('\t')
        rel, subj, obj, weight = data[0], data[1], data[2], float(data[3])

        subj, subj_pos = get_pos(subj)
        obj, obj_pos = get_pos(obj)

        if obj == subj:
            continue

        add_freq(concept = subj, freq = freq)
        add_freq(concept = obj, freq = freq)

    print('Finshed.')
    return freq


def find_lowfreq_words(opt, freq: Dict):
    lowfreq_words = []

    for concept, cnt in freq.items():
        if cnt <= opt.threshold:
            lowfreq_words.append(concept)

    json.dump(lowfreq_words, open(opt.output, 'w', encoding='utf8'), ensure_ascii=False, indent=4)
    print('Statistics:')
    print(f'Low frequency concepts with {opt.threshold} or less appearances in ConceptNet '
          f'make up {len(lowfreq_words) / len(freq)}% of the total concepts ({len(lowfreq_words)}/{len(freq)})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cpnet', type=str, default='./ConceptNet-en.csv', help='path to the english conceptnet')
    parser.add_argument('-output', type=str, default='./black_list.json', help='path to store the generated graph')
    parser.add_argument('-threshold', type=int, required=True, help='frequency threshold for pruning concepts (inclusive)')
    opt = parser.parse_args()
    freq = compute_freq(opt)
    find_lowfreq_words(opt, freq = freq)

