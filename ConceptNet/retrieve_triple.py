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
parser.add_argument('-cpnet', type=str, default='ConceptNet-en.pruned.txt', help='path to ConceptNet triples')
parser.add_argument('-output', type=str, default='./rough_retrieval.txt', help='file to store the output text')
opt = parser.parse_args()

with open(opt.cpnet, 'r', encoding='utf-8') as fin:
    cpnet = fin.readlines()


def stem(word: str) -> str:
    """
    Stem a single word
    """
    return stemmer.stem(word)


def match(entity_name: str, concept_name: str) -> bool:
    """
    Check whether a given concept matches with this entity.
    entity_name and concept_name should be the form after stemming.
    """
    entity = entity_name.split()
    concept = concept_name.split('_')

    if len(entity) == 1:  # single word
        return len(concept) == 1 and concept[0] == entity[0]

    # multiple words
    concept = set(concept)
    entity = set(entity)
    common_words = len(concept.intersection(entity))
    total_words = len(concept.union(entity))
    return common_words / total_words >= 0.5



def related(entity: str, line: List[str]) -> bool:
    """
    Check whether a given ConceptNet triple is related to this entity.
    """
    assert len(line) == 8
    stem_subj, stem_obj = line[1], line[4]
    subj_pos, obj_pos = line[3], line[6]

    if match(entity_name=entity, concept_name=stem_subj) and subj_pos in ['n', '-']:
        return True

    if match(entity_name=entity, concept_name=stem_obj) and obj_pos in ['n', '-']:
        return True

    return False


def search_triple(entity: str) -> List:
    global cpnet
    entity = entity.strip()
    stem_entity = ' '.join(map(stem, entity.split()))
    triple_list = []

    for line in cpnet:
        line = line.strip().split('\t')
        if related(entity = stem_entity, line = line):
            triple_list.append(line)

    return triple_list


def retrieve(datapath: str) -> (List, int):
    """
    Retrieve all possibly related triples from ConceptNet to the given entity. (rough retrieval)
    For entities with single
    Args:
        datapath - path to the input dataset
        fout - file object to store output
    """
    triple_cnt = 0
    result = []
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

    return result, triple_cnt


if __name__ == '__main__':
    result = []

    print('Dev')
    fout = open(opt.output, 'w', encoding='utf-8')  # write a new file
    dev_result, triple_cnt = retrieve(opt.dev)
    result.extend(dev_result)
    json.dump(result, fout, indent=4, ensure_ascii=False)
    fout.close()
    print(f'{len(dev_result)} data instances acquired.')
    print(f'Average number of ConceptNet triples found: {triple_cnt / len(dev_result)}')

    print('Test')
    fout = open(opt.output, 'w', encoding='utf-8')  # append to previous lines
    test_result, triple_cnt = retrieve(opt.test)
    result.extend(test_result)
    json.dump(result, fout, indent=4, ensure_ascii=False)
    fout.close()
    print(f'{len(test_result)} data instances acquired.')
    print(f'Average number of ConceptNet triples found: {triple_cnt / len(test_result)}')

    print('Train')
    fout = open(opt.output, 'w', encoding='utf-8')
    train_result, triple_cnt = retrieve(opt.train)
    result.extend(train_result)
    json.dump(result, fout, indent=4, ensure_ascii=False)
    fout.close()
    print(f'{len(train_result)} data instances acquired.')
    print(f'Average number of ConceptNet triples found: {triple_cnt / len(train_result)}')

