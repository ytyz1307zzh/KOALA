'''
 @Date  : 12/31/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import argparse
import json
from tqdm import tqdm
import time
import re

log_file = open('../logs/debug.log', 'w', encoding='utf8')


def find_word_with_pos(word):
    word_split = word.split("/")
    if len(word_split) > 5:
        word_split = word_split[:5]
    if word_split[-1] in ["n", "a", "v", "r"]:
        return '/'.join(word_split[-2:])
    else:
        return word_split[-1]

def extract_english(opt):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    """
    all_rel_types = []
    fout = open(opt.output, "w", encoding="utf8")

    start_time = time.time()
    fin = open(opt.cpnet, 'r', encoding="utf8")
    lines = fin.readlines()
    print(f'File read time: {time.time() - start_time:.2f}s')

    for line in tqdm(lines):
        ls = line.split('\t')

        # keep those triples with an English head concept and an English tail concept
        if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):

            rel = ls[1].split("/")[-1].lower()
            head = find_word_with_pos(ls[2]).lower()
            tail = find_word_with_pos(ls[3]).lower()

            if not re.sub(r'[/_\-]', '', head).isalpha():
                print(head, file=log_file)
                continue

            if not re.sub(r'[/_\-]', '', tail).isalpha():
                print(tail, file=log_file)
                continue

            all_rel_types.append(rel)
            data = json.loads(ls[4])
            fout.write("\t".join([rel, head, tail, str(data["weight"])]) + '\n')

    fin.close()
    fout.close()

    # calculate some statictics
    print('All relation types:')
    for type in set(all_rel_types):
        type_appear = all_rel_types.count(type)
        data_length = len(all_rel_types)
        print(f'{type}: {type_appear}/{data_length} ({(type_appear/data_length)*100:.2f}%)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cpnet', type=str, default='./ConceptNet.csv', help='the downloaded conceptnet file')
    parser.add_argument('-output', type=str, default='./ConceptNet-en.csv', help='the path to store output data')
    opt = parser.parse_args()

    print('Start processing...')
    start_time = time.time()
    extract_english(opt)
    print(f'Finished. Time Elapse: {time.time() - start_time:.2f}s')

