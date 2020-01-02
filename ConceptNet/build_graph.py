'''
 @Date  : 01/02/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import networkx as nx
from tqdm import tqdm
import spacy
import argparse
import re
from typing import Dict, List
nlp = spacy.load("en_core_web_sm", disable = ['parser', 'ner'])
log_file = open('../logs/debug.log', 'w', encoding='utf8')


def lemmatize(phrase: str) -> (List[str], str):
    phrase = re.sub('_', ' ', phrase)
    doc = nlp(phrase)
    lemma_list = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc]
    return '_'.join(lemma_list)


def get_relations(rel_filepath: str) -> Dict:
    """
    Get the valid relation list from rel_filepath
    """
    rel_dict = {}
    with open(rel_filepath, 'r', encoding='utf8') as fin:
        for line in fin:
            relation, direction = line.strip().split(': ')
            rel_dict[relation] = direction

    return rel_dict


def get_pos(concept: str) -> (str, str):
    """
    If the concept has specific pos, extract it
    """
    if concept.endswith('/n') or concept.endswith('/v') or concept.endswith('/a') or concept.endswith('/r'):
        return concept[:-2], concept[-1]
    else:
        return concept, '-'

def build_graph(opt):
    """
    Build a directional graph from csv for ConceptNet.
    Concepts are lemmatized before registering as nodes.
    """
    rel_dict = get_relations(opt.relation)
    fout = open(opt.output, 'w', encoding='utf8')
    with open(opt.cpnet, 'r', encoding='utf8') as fin:
        cp_lines = fin.readlines()

    print('Start transforming...')
    for line in tqdm(cp_lines):

        data = line.strip().split('\t')
        rel, subj, obj, weight = data[0], data[1], data[2], float(data[3])

        if rel not in rel_dict.keys():
            continue

        subj, subj_pos = get_pos(subj)
        obj, obj_pos = get_pos(obj)

        # subj = lemmatize(subj)
        # obj = lemmatize(obj)

        if subj == obj:
            print(line, file=log_file)
            continue

        if rel == 'relatedto' or rel == 'antonym':
            weight -= 0.3

        fout.write('\t'.join([rel, subj, subj_pos, obj, obj_pos, str(weight)]) + '\n')

    print('Finshed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cpnet', type=str, default='./ConceptNet-en.csv', help='path to the english conceptnet')
    parser.add_argument('-relation', type=str, default='./relation_direction.txt', help='file that specifies the valid relations')
    parser.add_argument('-output', type=str, default='./ConceptNet-en.pruned.txt', help='path to store the generated graph')
    opt = parser.parse_args()
    build_graph(opt)

