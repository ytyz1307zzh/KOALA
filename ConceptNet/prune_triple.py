'''
 @Date  : 01/03/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import json
from tqdm import tqdm
import argparse
import re
from typing import List, Dict, Set
import nltk
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may"]
from Stemmer import PorterStemmer
stemmer = PorterStemmer()


def stem(word: str) -> str:
    """
    Stem a single word
    """
    return stemmer.stem(word)


def read_relation(filename: str) -> Dict[str, str]:
    file = open(filename, 'r', encoding='utf8')
    rel_rules = {}

    for line in file:
        rule = line.strip().split(': ')
        relation, direction = rule
        rel_rules[relation] = direction

    return rel_rules


def extract_context(paragraph) -> Set[str]:
    """
    Acquire all content words from a paragraph.
    """
    paragraph = paragraph.lower().strip().split()
    return {word for word in paragraph if word not in nltk_stopwords}


def valid_direction(relation: str, direction: str, rel_rules: Dict[str, str]) -> bool:
    """
    Check the semantic role of the entity (subj or obj) is valid or not, according to the rel_rules
    """
    assert direction in ['left', 'right']
    rule = rel_rules[relation]
    assert rule in ['left', 'right', 'both']

    if rule == 'both':
        return True
    elif rule == direction:
        return True
    else:
        return False


def select_triple(entity: str, raw_triples: List[str], context_set: Set[str], rel_rules: Dict[str, str]):
    """
    Select related triples from the rough retrieval set.
    Args:
        entity - entity name, may contain a semicolon delimiter.
        context_set - the content words in the context (paragraph + topic)
        rel_rules - selection rules for each relation type (subj only, or both subj and obj?)
    """
    selected_triples = []
    stem_context = set(map(stem, context_set))

    for line in raw_triples:

        triple = line.strip().split(', ')
        assert len(triple) == 9

        direction = triple[-1]  # LEFT or RIGHT
        relation = triple[0]  # relation type

        # if the semantic role of the entity (subj or obj) does not match, skip this
        if not valid_direction(relation = relation, direction = direction.lower(), rel_rules = rel_rules):
            continue

        # find the neighbor concept
        if direction == 'LEFT':
            neighbor = set(triple[4].strip().split('_'))
        elif direction == 'RIGHT':
            neighbor = set(triple[1].strip().split('_'))

        if neighbor.intersection(stem_context):
            selected_triples.append(line)

    return selected_triples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./rough_retrieval.txt', help='path to the english conceptnet')
    parser.add_argument('-output', type=str, default='./retrieval.txt', help='path to store the generated graph')
    parser.add_argument('-relation', type=str, default='./relation_direction.txt', help='path to the relation rules')
    opt = parser.parse_args()

    data = json.load(open(opt.input, 'r', encoding='utf8'))
    rel_rules = read_relation(opt.relation)
    result = []

    for instance in data:
        para_id = instance['id']
        entity = instance['entity']
        paragraph = instance['paragraph']
        topic = instance['topic']
        raw_triples = instance['cpnet']

        context_set = extract_context(paragraph)
        context_set.union(extract_context(topic))

        selected_triples = select_triple(entity = entity, raw_triples = raw_triples, context_set = context_set, rel_rules = rel_rules)
        print(f'Triples before selection: {len(raw_triples)}, after selection: {len(selected_triples)}')

        result.append({'id': para_id,
                     'entity': entity,
                     'topic': topic,
                     'paragraph': paragraph,
                     'cpnet': selected_triples
                     })

    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'{len(result)} instances finished.')
