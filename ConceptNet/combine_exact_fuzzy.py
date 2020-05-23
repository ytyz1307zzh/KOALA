'''
 @Date  : 01/12/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import json
import argparse
from tqdm import tqdm
import os
import time
from typing import List

less_cnt = 0


def add_bert_identifier(triple: str) -> str:
    """
    Identify fuzzy-match triples by adding identifier "BERT", in correspondance to "RELEVANCE".
    """
    fields = triple.strip().split(', ')
    assert len(fields) == 12
    fields.insert(9, 'BERT')
    return ', '.join(fields)


def check_duplicate(triple: str, result: List[str]) -> bool:
    """
    Check whether there are duplicate triples from exact match and fuzzy match
    """
    fields = triple.strip().split(', ')
    assert len(fields) == 13
    assert fields[9] == 'BERT'
    psuedo_triple = fields[:11]
    psuedo_triple[9] = 'RELEVANCE'
    return ', '.join(psuedo_triple) in result


def combine_retrieval(bert_triples: List[str], rule_triples: List[str], max_num: int,
                      para_id: int, entity: str) -> (List[str], int, int):
    """
    Combine two sources of retrieved triples.
    Rule: exact-match triples in rule_triples come first, the remainings are fuzzy-match ones.
    """
    bert_len = len(bert_triples)
    rule_len = len(rule_triples)
    assert bert_len <= max_num, rule_len <= max_num
    if bert_len != rule_len:
        print(f'[INFO] Bert-based triples and rule-based triples have different lengths at Paragraph #{para_id} - {entity}')

    result = []
    for i in range(rule_len):
        triple = rule_triples[i].strip().split(', ')
        assert len(triple) == 11
        source = triple[9]  # RELEVANCE or SCORE
        assert source in ['RELEVANCE', 'SCORE']

        if source == 'RELEVANCE':
            result.append(rule_triples[i])

    bert_cnt, rule_cnt = 0, len(result)
    if len(result) == max_num:  # if relevance-based triples already make up to max_num, then return
        return result, bert_cnt, rule_cnt

    bert_triples = list(map(add_bert_identifier, bert_triples))
    for j in range(bert_len):
        # if this one has been retrieved in rule-based retrieval, skip it
        if check_duplicate(triple=bert_triples[j], result=result):
            continue
        result.append(bert_triples[j])
        bert_cnt += 1
        if len(result) == max_num:
            break

    global less_cnt
    if len(result) < max_num:
        print(f'[WARNING] Number of retrieved triples is less than {max_num} at Paragraph #{para_id} - {entity}')
        less_cnt += 1

    assert len(result) <= max_num
    return result, bert_cnt, rule_cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-rule', type=str, default='./result/retrieval_rule.json',
                        help='file containing retrieved triple using rule-based method')
    parser.add_argument('-bert', type=str, default='./result/retrieval_bert_word.json',
                        help='file containing retrieved triple using bert embedding')
    parser.add_argument('-output', type=str, default='./result/retrieval.json')
    parser.add_argument('-max_num', type=int, default=10, help='max number of retrieved triples')
    opt = parser.parse_args()

    bert_data = json.load(open(opt.bert, 'r', encoding='utf8'))
    rule_data = json.load(open(opt.rule, 'r', encoding='utf8'))
    assert len(bert_data) == len(rule_data)
    data_len = len(bert_data)

    result = []
    total_bert_cnt = 0
    total_rule_cnt = 0
    for i in tqdm(range(data_len)):
        bert_inst = bert_data[i]
        rule_inst = rule_data[i]

        para_id = bert_inst['id']
        entity = bert_inst['entity']
        topic = bert_inst['topic']
        paragraph = bert_inst['paragraph']
        prompt = bert_inst['prompt']

        bert_triples = bert_inst['cpnet']
        rule_triples = rule_inst['cpnet']

        combined_triples, bert_cnt, rule_cnt = combine_retrieval(bert_triples=bert_triples, rule_triples=rule_triples,
                                                                 max_num=opt.max_num, para_id=para_id, entity=entity)
        total_bert_cnt += bert_cnt
        total_rule_cnt += rule_cnt

        result.append({'id': para_id,
                       'entity': entity,
                       'topic': topic,
                       'prompt': prompt,
                       'paragraph': paragraph,
                       'cpnet': combined_triples
                       })

    json.dump(result, open(opt.output, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f'Total instances: {data_len}')
    print(f'Instances with less than {opt.max_num} ConceptNet triples collected: {less_cnt} ({(less_cnt / data_len) * 100:.2f}%)')
    print(f'Average number of BERT-retrieved triples: {total_bert_cnt / data_len}')
    print(f'Average number of rule-retrieved triples: {total_rule_cnt / data_len}')
    print('Finished.')


if __name__ == '__main__':
    main()

