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
from spacy.lang.en import STOP_WORDS
STOP_WORDS = set(STOP_WORDS) - {'bottom', 'serious', 'top', 'alone', 'around', 'used', 'behind', 'side', 'mine', 'well'}
from Stemmer import PorterStemmer
stemmer = PorterStemmer()
blacklist = {"uk", "us", "take", "make", "object", "person", "people"}


def lemmatize(phrase: str) -> str:
    phrase = re.sub('_', ' ', phrase)
    doc = nlp(phrase)
    lemma_list = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc]
    return '_'.join(lemma_list)


def stem(phrase: str):
    stem_list = []
    for token in phrase.split('_'):
        stem_list.append(stemmer.stem(token))
    return '_'.join(stem_list)


def remove_stopword(phrase: str) -> str:
    phrase = phrase.lower().strip().split('_')
    word_list = [word for word in phrase if word not in STOP_WORDS and word.isalpha()]
    return '_'.join(word_list)


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
    with open(opt.cpnet, 'r', encoding='utf8') as fin:
        cp_lines = fin.readlines()

    fout = open(opt.output, 'w', encoding='utf8')
    print('Start transforming...')
    for line in tqdm(cp_lines):

        data = line.strip().split('\t')
        rel, subj, obj, weight = data[0], data[1], data[2], float(data[3])

        if rel not in rel_dict.keys():
            continue

        subj, subj_pos = get_pos(subj)
        obj, obj_pos = get_pos(obj)

        if subj in blacklist or obj in blacklist:
            continue

        # subj = lemmatize(subj)
        # obj = lemmatize(obj)

        stem_subj = stem(remove_stopword(subj))
        stem_obj = stem(remove_stopword(obj))

        if stem_subj == stem_obj:
            continue

        if rel == 'relatedto' or rel == 'antonym':
            weight -= 0.3

        if rel == 'atlocation':
            weight += 0.5

        fout.write('\t'.join([rel, stem_subj, subj, subj_pos, stem_obj, obj, obj_pos, str(weight)]) + '\n')

    print('Finshed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-cpnet', type=str, default='./ConceptNet-en.csv', help='path to the english conceptnet')
    parser.add_argument('-relation', type=str, default='./relation_direction.txt', help='file that specifies the valid relations')
    parser.add_argument('-output', type=str, default='./ConceptNet-en.pruned.csv', help='path to store the generated graph')
    parser.add_argument('-override', default=False, action='store_true', help='if specified, -output will be the same with -cpnet')
    opt = parser.parse_args()

    if opt.override:
        opt.output = opt.cpnet
    build_graph(opt)

