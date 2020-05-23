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
# import nltk
# nltk.download('stopwords')
# nltk_stopwords = nltk.corpus.stopwords.words('english')
# nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may"]
from spacy.lang.en import STOP_WORDS
STOP_WORDS = set(STOP_WORDS) - {'bottom', 'serious', 'top', 'alone', 'around', 'used', 'behind', 'side', 'mine', 'well'}
from Stemmer import PorterStemmer
stemmer = PorterStemmer()


def stem(word: str) -> str:
    """
    Stem a single word
    """
    word = word.lower().strip()
    return stemmer.stem(word)


def get_weight(line: str) -> float:
    triple = line.strip().split(', ')
    assert len(triple) == 9
    weight = float(triple[7])
    return weight


def get_relation(line: str) -> str:
    triple = line.strip().split(', ')
    assert len(triple) == 9
    relation = triple[0]
    return relation


def read_relation(filename: str) -> Dict[str, str]:
    file = open(filename, 'r', encoding='utf8')
    rel_rules = {}

    for line in file:
        rule = line.strip().split(': ')
        relation, direction = rule
        rel_rules[relation] = direction

    return rel_rules


def read_transform(filename: str) -> Dict[str, str]:
    file = open(filename, 'r', encoding='utf8')
    trans_rules = {}

    for line in file:
        relation, sentence = line.strip().split(': ')
        trans_rules[relation] = sentence.strip()

    return trans_rules


def remove_stopword(paragraph: str) -> Set[str]:
    """
    Acquire all content words from a paragraph.
    """
    paragraph = paragraph.lower().strip().split()
    return {word for word in paragraph if word not in STOP_WORDS and word.isalpha()}


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


def in_context(concept: Set, context: Set) -> bool:
    """
    Score: (words in both concept and context) / (words in concept)
    """
    score = len(concept.intersection(context)) / len(concept)
    return score >= 0.5


def triple2sent(raw_triples: List[str], trans_rules: Dict[str, str]):
    """
    Turn the triples to natural language sentences.
    """
    result = []

    for line in raw_triples:

        triple = line.strip().split(', ')
        assert len(triple) == 10

        relation = triple[0]
        subj = triple[2]
        obj = triple[5]

        # turn the delimiter from _ to space
        subj = ' '.join(subj.split('_'))
        obj = ' '.join(obj.split('_'))

        sentence = trans_rules[relation]
        sentence = re.sub('A', subj, sentence)
        sentence = re.sub('B', obj, sentence)

        triple.append(sentence)
        result.append(', '.join(triple))

    return result


def select_triple(entity: str, raw_triples: List[str], context_set: Set[str],
                  rel_rules: Dict[str, str], max: int) -> (List[str], int, int):
    """
    Select related triples from the rough retrieval set.
    Args:
        entity - entity name, may contain a semicolon delimiter.
        context_set - the content words in the context (paragraph + topic)
        rel_rules - selection rules for each relation type (subj only, or both subj and obj?)
    """
    triples_by_score = []
    triples_by_relevance = []

    entity_list = entity.split(';')
    entity_set = set()
    for ent in entity_list:
        entity_set = entity_set.union(set(map(stem, ent.split())))

    stem_context = set(map(stem, context_set)) - entity_set

    for line in raw_triples:

        triple = line.strip().split(', ')
        assert len(triple) == 9

        direction = triple[-1]  # LEFT or RIGHT
        relation = triple[0]  # relation type

        # if the semantic role of the entity (subj or obj) does not match, skip this
        if not valid_direction(relation = relation, direction = direction.lower(), rel_rules = rel_rules):
            continue

        triples_by_score.append(line)

        # find the neighbor concept
        if direction == 'LEFT':
            neighbor = set(triple[4].strip().split('_'))
        elif direction == 'RIGHT':
            neighbor = set(triple[1].strip().split('_'))

        if in_context(concept = neighbor, context = stem_context):
            triples_by_relevance.append(line)

    # retrieve at most max/2 relevance-based triples, the others are filled with score-based triples
    triples_by_relevance = [t for t in triples_by_relevance if get_weight(t) >= 1.0]
    triples_by_relevance = sorted(triples_by_relevance, key=lambda x: get_relation(x) != 'relatedto', reverse=True)
    triples_by_relevance = sorted(triples_by_relevance, key=get_weight, reverse=True)
    if len(triples_by_relevance) > max:
        triples_by_relevance = triples_by_relevance[:max]

    triples_by_score = list(set(triples_by_score) - set(triples_by_relevance))
    triples_by_score = sorted(triples_by_score, key = lambda x: get_relation(x) != 'relatedto', reverse = True)
    triples_by_score = sorted(triples_by_score, key = get_weight, reverse = True)
    triples_by_score = triples_by_score[:(max - len(triples_by_relevance))]

    triples_by_score = [t + ', SCORE' for t in triples_by_score]
    triples_by_relevance = [t + ', RELEVANCE' for t in triples_by_relevance]

    result = triples_by_relevance + triples_by_score
    assert len(set(result)) == len(result)  # no duplicate items

    return result, len(triples_by_relevance), len(triples_by_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./rough_retrieval.json', help='path to the english conceptnet')
    parser.add_argument('-output', type=str, default='./retrieval.json', help='path to store the generated graph')
    parser.add_argument('-relation', type=str, default='./relation_direction.txt', help='path to the relation rules')
    parser.add_argument('-transform', type=str, default='./triple2sent.txt',
                        help='path to the file that describes the rules to transform triples into natural language')
    parser.add_argument('-max', type=int, default=10, help='how many triples to collect')
    opt = parser.parse_args()

    data = json.load(open(opt.input, 'r', encoding='utf8'))
    rel_rules = read_relation(opt.relation)
    trans_rules = read_transform(opt.transform)
    result = []
    less_cnt, total_relevance, total_score = 0, 0, 0

    for instance in tqdm(data):
        para_id = instance['id']
        entity = instance['entity']
        paragraph = instance['paragraph']
        topic = instance['topic']
        prompt = instance['prompt']
        raw_triples = instance['cpnet']

        context_set = remove_stopword(prompt + ' ' + paragraph)

        # raw_triples may contain repetitive fields (multiple entities)
        selected_triples, num_relevance, num_score = select_triple(entity = entity, raw_triples = list(set(raw_triples)),
                                                                   context_set = context_set, rel_rules = rel_rules, max = opt.max)

        total_relevance += num_relevance
        total_score += num_score

        if len(selected_triples) < opt.max:
            less_cnt += 1

        selected_triples = triple2sent(raw_triples = selected_triples, trans_rules = trans_rules)

        result.append({'id': para_id,
                       'entity': entity,
                       'topic': topic,
                       'prompt': prompt,
                       'paragraph': paragraph,
                       'cpnet': selected_triples
                       })

    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

    total_instances = len(result)
    print(f'Total instances: {total_instances}')
    print(f'Instances with less than {opt.max} ConceptNet triples collected: {less_cnt} ({(less_cnt/total_instances)*100:.2f}%)')
    print(f'Average number of relevance-based triples: {total_relevance / total_instances:.2f}')
    print(f'Average number of score-based triples: {total_score / total_instances:.2f}')
    print(f'{len(result)} instances finished.')
