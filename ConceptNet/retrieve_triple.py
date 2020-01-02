'''
 @Date  : 01/02/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import json
from tqdm import tqdm
import spacy
import argparse
import re
from typing import Dict, List
nlp = spacy.load("en_core_web_sm", disable = ['parser', 'ner'])
import nltk
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may"]
from Stemmer import PorterStemmer
stemmer = PorterStemmer()

parser = argparse.ArgumentParser()
parser.add_argument('-train', type=str, default='../data/train.json', help='path to the training set')
parser.add_argument('-dev', type=str, default='../data/dev.json', help='path to the dev set')
parser.add_argument('-test', type=str, default='../data/test.json', help='path to the test set')
parser.add_argument('-cpnet', type=str, default='ConceptNet-en.csv', help='path to ConceptNet triples')
parser.add_argument('-output', type=str, default='./rough_retrieval.txt', help='file to store the output text')
opt = parser.parse_args()

with open(opt.cpnet, 'r', encoding='utf-8') as fin:
    cpnet = fin.readlines()

result = []
triple_cnt = 0


def stem(word: str) -> str:
    """
    Stem a single word
    """
    return stemmer.stem(word)


def match(entity_name: str, concept_name: str) -> bool:
    """
    Check whether a given concept matches with this entity.
    """
    entity = entity_name.split()
    concept = concept_name.split('_')

    if len(entity) == 1:  # single word
        return len(concept) == 1 and stem(concept[0]) == stem(entity[0])

    # multiple words
    concept = set(map(stem, concept))
    entity = set(map(stem, entity))
    common_words = len(concept.intersection(entity))
    total_words = len(concept.union(entity))
    return common_words / total_words >= 0.5



def related(entity: str, line: List[str]) -> bool:
    """
    Check whether a given ConceptNet triple is related to this entity.
    """
    assert len(line) == 6
    subj, obj = line[1], line[3]
    subj_pos, obj_pos = line[2], line[4]

    if match(entity, subj) and subj_pos in ['n', '-']:
        return True

    if match(entity, obj) and obj_pos in ['n', '-']:
        return True

    return False


def search_triple(entity: str) -> List:
    global cpnet
    entity = entity.strip()
    triple_list = []

    for line in cpnet:
        line = line.strip().split('\t')
        if related(entity = entity, line = line):
            triple_list.append(line)

    return triple_list


def retrieve(datapath: str):
    """
    Retrieve all possibly related triples from ConceptNet to the given entity. (rough retrieval)
    For entities with single
    Args:
        datapath - path to the input dataset
        fout - file object to store output
    """
    global result, triple_cnt
    dataset = json.load(open(datapath, 'r', encoding='utf8'))

    for instance in tqdm(dataset):
        para_id = instance['id']
        entity_name = instance['entity']
        topic = instance['topic']
        paragraph = instance['paragraph']

        triples = []
        for ent in entity_name.split(';'):
            triples.extend(search_triple(ent))
        triple_cnt += len(triples)

        result.append({'id': para_id,
                       'entity': entity_name,
                       'topic': topic,
                       'paragraph': paragraph,
                       'cpnet': triples
                       })


if __name__ == '__main__':
    print('Dev')
    retrieve(opt.dev)
    print('Test')
    retrieve(opt.test)
    print('Train')
    retrieve(opt.train)

    print(f'{len(result)} data instances acquired.')
    print(f'Average number of ConceptNet triples found: {triple_cnt / len(result)}')
    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
