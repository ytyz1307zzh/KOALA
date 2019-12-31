'''
 @Date  : 12/02/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

"""
Reads the raw csv files downloaded from the Propara website, then create JSON files which contain
lists of instances in train, dev and test

JSON format: a list of instances
Instance
    |____topic
    |____prompt
    |____paragraph id
    |____paragraph (string)
    |____entity (string)
    |____gold state change sequence (list of labels)
    |____number of words (tokens) in the paragraph
    |____number of sentences in the paragraph
    |____number of location candidates
    |____list of location candidates
    |____gold locations (list of strings, len = sent + 1)
    |____list of sentences
               |____sentence id
               |____number of words
               |____sentence (string)
               |____entity mention position (list of indices)
               |____verb mention position (list of indices)
               |____list of location candidates
                            |____location mention position (list of indices)
                                (list length equal to number of location candidates)
"""

from typing import Dict, List, Tuple, Set
import pandas as pd
import argparse
import json
import os
import time
import re
pd.set_option('display.max_columns', 50)
total_paras = 0  # should equal to 488 after read_paragraph

import spacy
nlp = spacy.load("en_core_web_sm", disable = ['parser', 'ner'])

from flair.data import Sentence
from flair.models import SequenceTagger
import flair
pos_tagger = SequenceTagger.load('pos')


def tokenize(paragraph: str) -> (str, int):
    """
    Change the paragraph to lower case and tokenize it!
    """
    paragraph = re.sub(' +', ' ', paragraph)  # remove redundant spaces in some sentences.
    para_doc = nlp(paragraph.lower())  # create a SpaCy Doc instance for paragraph
    tokens_list = [token.text for token in para_doc]
    return ' '.join(tokens_list), len(tokens_list)


def lemmatize(paragraph: str) -> (List[str], str):
    """
    Reads a paragraph/sentence/phrase/word and lemmatize it!
    """
    if paragraph == '-' or paragraph == '?':
        return None, paragraph
    para_doc = nlp(paragraph)
    lemma_list = [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in para_doc]
    return lemma_list, ' '.join(lemma_list)


# TODO: Maybe we shouldn't perform lemmatization to location candidates for the test set
#       in order to generate raw spans in the paragraph while filling the grids.
#       (candidate masks are still computed after masking both the candidate and the paragraph)
def find_loc_candidate(paragraph: flair.data.Sentence) -> Set[str]:
    """
    paragraph: the paragraph after tokenization and lower-case transformation
    return: the location candidates found in this paragraph
    """
    pos_tagger.predict(paragraph)
    pos_list = [(token.text, token.get_tag('pos').value) for token in paragraph]
    loc_list = []

    # extract nouns (including 'noun + noun' phrases)
    for i in range(len(pos_list)):
        if pos_list[i][1] == 'NOUN':
            candidate = pos_list[i][0]
            for k in range(1, i+1):
                if pos_list[i-k][1] == 'ADJ':
                    candidate = pos_list[i-k][0] + ' ' + candidate
                elif pos_list[i-k][1] == 'NOUN':
                    loc_list.append(candidate)
                    candidate = pos_list[i-k][0] + ' ' + candidate
                else:
                    break
            loc_list.append(candidate)

    # extract 'noun + and/or + noun' phrase
    for i in range(2, len(pos_list)):
        if pos_list[i][1] == 'NOUN' \
            and (pos_list[i-1][0] == 'and' or pos_list[i-1][0] == 'or') \
                and pos_list[i-2][1] == 'NOUN':
            loc_list.append(pos_list[i-2][0] + ' ' + pos_list[i-1][0] + ' ' + pos_list[i][0])
    
    # lemmatization
    for i in range(len(loc_list)):
        _, location = lemmatize(loc_list[i])
        loc_list[i] = location
    
    return set(loc_list)


def find_mention(paragraph: List[str], phrase: str, norm: bool) -> List:
    """
    Judge whether a phrase is a span of the paragraph (or sentence) and return the span
    norm: whether the sentence should be normalized first
    """
    phrase = phrase.strip().split()
    phrase_len = len(phrase)
    span_list = []

    # perform lemmatization on both the paragraph and the phrase
    if norm:
        paragraph, _ = lemmatize(' '.join(paragraph))
        phrase, _ = lemmatize(' '.join(phrase))

    for i in range(0, len(paragraph) - phrase_len):
        sub_para = paragraph[i: i+phrase_len]
        if sub_para == phrase:
            span_list.extend(list(range(i, i+phrase_len)))
    return span_list


def log_existence(paragraph: str, para_id: int, entity: str, loc_seq: List[str], log_file):
    """
    Record the entities and locations that does not match any span in the paragraph.
    """
    entity_list = re.split('; |;', entity)
    paragraph = paragraph.strip().split()
    for ent in entity_list:
        if not find_mention(paragraph, ent, norm = False) and not find_mention(paragraph, ent, norm = True):
            print(f'[WARNING] Paragraph {para_id}: entity "{ent}" is not a span in paragraph.', file=log_file)
    
    for loc in loc_seq:
        if loc == '-' or loc == '?':
            continue
        if not find_mention(paragraph, loc, norm = True):
            print(f'[WARNING] Paragraph {para_id}: location "{loc}" is not a span in paragraph.', file=log_file)


def get_entity_mask(sentence: str, entity: str, pad_bef_len: int, pad_aft_len: int) -> List[int]:
    """
    return the masked vector pertaining to a certain entity in the paragraph
    """
    sentence = sentence.strip().split()
    sent_len = len(sentence)
    entity_list = re.split('; |;', entity)
    span_list = []
    for ent_name in entity_list:
        span_list.extend(find_mention(sentence, ent_name, norm = False) or find_mention(sentence, ent_name, norm = True))
    
    entity_mask = [1 if i in span_list else 0 for i in range(sent_len)]
    padding_before = [0 for _ in range(pad_bef_len)]
    padding_after = [0 for _ in range(pad_aft_len)]

    return padding_before + entity_mask + padding_after


def get_verb_mask(sentence: str, pad_bef_len: int, pad_aft_len: int) -> List[int]:
    """
    return the masked vector pertaining to the verb in the sentence
    """
    sentence = Sentence(sentence)
    pos_tagger.predict(sentence)
    sent_len = len(sentence)
    pos_list = [(token.text, token.get_tag('pos').value) for token in sentence]
    span_list = [i for i in range(sent_len) if pos_list[i][1] == 'VERB']
    
    verb_mask = [1 if i in span_list else 0 for i in range(sent_len)]
    padding_before = [0 for _ in range(pad_bef_len)]
    padding_after = [0 for _ in range(pad_aft_len)]

    return padding_before + verb_mask + padding_after


def get_location_mask(sentence: str, location: str, pad_bef_len: int, pad_aft_len: int) -> List[int]:
    """
    return the masked vector pertaining to a certain location in the paragraph
    """
    sentence = sentence.strip().split()
    sent_len = len(sentence)
    span_list = find_mention(sentence, location, norm = True)
    
    loc_mask = [1 if i in span_list else 0 for i in range(sent_len)]
    padding_before = [0 for _ in range(pad_bef_len)]
    padding_after = [0 for _ in range(pad_aft_len)]

    return padding_before + loc_mask + padding_after


def compute_state_change_seq(gold_loc_seq: List[str]) -> List[str]:
    """
    Compute the state change sequence for the certain entity.
    Note that the gold location sequence contains an 'initial state'.
    State change labels: O_C, O_D, E, M, C, D
    """
    num_states = len(gold_loc_seq)
    # whether the entity has been created. (if exists from the beginning, then it should be True)
    create = False if gold_loc_seq[0] == '-' else True
    gold_state_seq = []

    for i in range(1, num_states):

        if gold_loc_seq[i] == '-':  # could be O_C, O_D or D
            if create == True and gold_loc_seq[i-1] == '-':
                gold_state_seq.append('O_D')
            elif create == True and gold_loc_seq[i-1] != '-':
                gold_state_seq.append('D')
            else:
                gold_state_seq.append('O_C')

        elif gold_loc_seq[i] == gold_loc_seq[i-1]:
            gold_state_seq.append('E')

        else:  # location change, could be C or M
            if gold_loc_seq[i-1] == '-':
                create = True
                gold_state_seq.append('C')
            else:
                gold_state_seq.append('M')
    
    assert len(gold_state_seq) == len(gold_loc_seq) - 1
    
    return gold_state_seq


def read_paragraph(filename: str) -> Dict[int, Dict]:

    csv_data = pd.read_csv(filename)
    paragraph_result = {}
    max_sent = len(csv_data.columns) - 3  # should equal to 10 in this case

    for _, row in csv_data.iterrows():
        para_id = int(row['Paragraph ID'])
        topic = row['Topic']
        prompt = row['Prompt']
        sent_list = []

        for i in range(1, max_sent + 1):
            sent = row[f'Sentence{i}']
            if pd.isna(sent):
                break
            sent_list.append(sent)

        text = ' '.join(sent_list)
        paragraph_result[para_id] = {'id': para_id,
                                     'topic': topic,
                                     'prompt': prompt,
                                     'paragraph': text,
                                     'total_sents': len(sent_list)}
    
    total_paras = len(paragraph_result)
    print(f'Paragraphs read: {total_paras}')
    return paragraph_result


def read_paragraph_from_sentences(csv_data: pd.DataFrame, begin_row_index: int, total_sents: int) -> str:
    """
    Read the paragraph from State_change_annotations.csv.
    This is because the paragraph in this file and the original Paragraphs.csv may be different and will cause problems.
    """
    row_index = begin_row_index + 3  # row index of the first sentence
    sent_list = []

    for i in range(total_sents):  # read each sentence
        row = csv_data.iloc[row_index]
        assert row['sent_id'] == f'event{i + 1}'
        sentence = row['sentence']
        sent_list.append(sentence)
        row_index += 2

    return ' '.join(sent_list)


def read_annotation(filename: str, paragraph_result: Dict[int, Dict],
                    log_file, test: bool) -> List[Dict]:
    """
    1. read csv
    2. get the entities
    3. tokenize the paragraph and change to lower case
    3. extract location candidates
    4. for each entity, create an instance for it
    5. read the entity's initial state
    5. read each sentence, give it an ID
    6. compute entity mask (length of mask vector = length of paragraph)
    7. extract the nearest verb to the entity, compute verb mask
    8. for each location candidate, compute location mask
    9. read entity's state at current timestep
    10. for the train/dev sets, if gold location is not extracted in step 3,
        add it to the candidate set (except for '-' and '?'). Back to step 6
    11. reading ends, compute the number of sentences
    12. get the number of location candidates
    13. infer the gold state change sequence
    """

    data_instances = []
    column_names = ['para_id', 'sent_id', 'sentence', 'ent1', 'ent2', 'ent3',
                    'ent4', 'ent5', 'ent6', 'ent7', 'ent8']
    max_entity = 8

    csv_data = pd.read_csv(filename, header = None, names = column_names)
    row_index = 0
    para_index = 0

    # variables for computing the accuracy of location prediction
    total_loc_cnt = 0
    total_err_cnt = 0

    start_time = time.time()

    while True:

        row = csv_data.iloc[row_index]
        if pd.isna(row['para_id']):  # skip empty lines
            row_index += 1
            continue

        para_id = int(row['para_id'])
        if para_id not in paragraph_result:  # keep the dataset split
            row_index += 1
            continue
        
        # the number of lines we need to read is relevant to 
        # the number of sentences in this paragraph
        total_sents = paragraph_result[para_id]['total_sents']
        total_lines = 2 * total_sents + 3
        begin_row_index = row_index  # first line of this paragraph in csv
        end_row_index = row_index + total_lines - 1  # last line

        # tokenize, lower cased
        raw_paragraph = read_paragraph_from_sentences(csv_data = csv_data, begin_row_index = begin_row_index, total_sents = total_sents)
        paragraph, total_tokens = tokenize(raw_paragraph)
        prompt, _ = tokenize(paragraph_result[para_id]['prompt'])

        # find location candidates
        loc_cand_set = find_loc_candidate(Sentence(paragraph))
        print(f'Paragraph {para_id}: \nLocation candidate set: ', loc_cand_set, file=log_file)

        # process data in this paragraph
        # first, figure out how many entities it has
        entity_list = []
        for i in range(1, max_entity + 1):
            entity_name = row[f'ent{i}']
            if pd.isna(entity_name):
                break
            entity_list.append(entity_name)
        
        total_entities = len(entity_list)
        verb_mention_per_sent = [None for _ in range(total_sents)]

        # sets for computing the accuracy of location prediction
        total_loc_set = set()
        total_err_set = set()

        for i in range(total_entities):
            entity_name = entity_list[i]

            instance = {'id': para_id,
                        'topic': paragraph_result[para_id]['topic'],
                        'prompt': prompt,
                        'paragraph': paragraph,
                        'total_tokens': total_tokens,
                        'total_sents': total_sents,
                        'entity': entity_name}
            gold_loc_seq = []  # list of gold locations
            sentence_list = []
            sentence_concat = []

            # read initial state, skip the prompt line
            row_index += 2
            row = csv_data.iloc[row_index]
            assert row['sent_id'] == 'state1'
            _, gold_location = lemmatize(row[f'ent{i+1}'])
            gold_loc_seq.append(gold_location)

            # for each sentence, read the sentence and the entity location
            for j in range(total_sents):

                # read sentence
                row_index += 1
                row = csv_data.iloc[row_index]
                assert row['sent_id'] == f'event{j+1}'
                sentence, num_tokens_in_sent = tokenize(row['sentence'])
                sentence_concat.append(sentence)
                sent_id = j + 1

                # read gold state
                row_index += 1
                row = csv_data.iloc[row_index]
                assert row['sent_id'] == f'state{j+2}'
                _, gold_location = lemmatize(row[f'ent{i+1}'])
                gold_loc_seq.append(gold_location)

                if gold_location != '-' and gold_location != '?':
                    total_loc_set.add(gold_location)


                # whether the gold location is in the candidates (training only)
                if gold_location not in loc_cand_set \
                    and gold_location != '-' and gold_location != '?':
                    if not test:
                        loc_cand_set.add(gold_location)
                    total_err_set.add(gold_location)
                    print(f'[INFO] Paragraph {para_id}: gold location "{gold_location}" not included in candidate set.',
                         file=log_file)

                sentence_list.append({'id': sent_id, 
                                      'sentence': sentence, 
                                      'total_tokens': num_tokens_in_sent})
            
            assert len(sentence_list) == total_sents
            loc_cand_list = list(loc_cand_set)
            total_loc_candidates = len(loc_cand_list)
            # record the entities and locations that does not match any span in the paragraph
            entity_name, _ = tokenize(entity_name)
            log_existence(paragraph, para_id, entity_name, gold_loc_seq, log_file)
            
            words_read = 0  # how many words have been read
            for j in range(total_sents):

                sent_dict = sentence_list[j]
                sentence = sent_dict['sentence']
                num_tokens_in_sent = sent_dict['total_tokens']

                # compute the masks
                entity_mask = get_entity_mask(sentence, entity_name, words_read, total_tokens - words_read)
                entity_mention = [idx for idx in range(len(entity_mask)) if entity_mask[idx] == 1]

                if not verb_mention_per_sent[j]:
                    verb_mask = get_verb_mask(sentence, words_read, total_tokens - words_read)
                    assert len(entity_mask) == len(verb_mask)
                    verb_mention = [idx for idx in range(len(verb_mask)) if verb_mask[idx] == 1]
                    verb_mention_per_sent[j] = verb_mention
                else:
                    verb_mention = verb_mention_per_sent[j]
                
                loc_mention_list = []
                for loc_candidate in loc_cand_list:
                    loc_mask = get_location_mask(sentence, loc_candidate, words_read, total_tokens - words_read)
                    assert len(entity_mask) == len(loc_mask)
                    loc_mention = [idx for idx in range(len(loc_mask)) if loc_mask[idx] == 1]
                    loc_mention_list.append(loc_mention)

                sent_dict['entity_mention'] = entity_mention
                sent_dict['verb_mention'] = verb_mention
                sent_dict['loc_mention_list'] = loc_mention_list
                sentence_list[j] = sent_dict
                words_read += num_tokens_in_sent

            assert words_read == total_tokens  # length of each sentence should sum up to length of the paragraph
            assert len(gold_loc_seq) == len(sentence_list) + 1
            instance['sentence_list'] = sentence_list
            instance['loc_cand_list'] = loc_cand_list
            instance['total_loc_candidates'] = total_loc_candidates
            instance['gold_loc_seq'] = gold_loc_seq
            instance['gold_state_seq'] = compute_state_change_seq(gold_loc_seq)
            # print(instance)
            assert paragraph == ' '.join(sentence_concat), f'at paragraph #{para_id}'

            # pointer backward, construct instance for next entity
            row_index = begin_row_index
            data_instances.append(instance)

        total_loc_cnt += len(total_loc_set)
        total_err_cnt += len(total_err_set)
        # print(total_loc_set)
        # print(total_err_set)


        row_index = end_row_index + 1
        para_index += 1

        if para_index % 10 == 0:
            end_time = time.time()
            print(f'[INFO] {para_index} paragraphs processed. Time elapse: {end_time - start_time}s')
        if para_index >= len(paragraph_result):
            end_time = time.time()
            print(f'[INFO] All {para_index} paragraphs processed. Time elapse: {end_time - start_time}s')
            break
    
    # compute accuracy of location prediction
    loc_accuracy = 1 - total_err_cnt / total_loc_cnt
    print(f'[DATA] Recall of location prediction: {loc_accuracy} ({total_loc_cnt - total_err_cnt}/{total_loc_cnt})')

    return data_instances


def read_split(filename: str, paragraph_result: Dict[int, Dict]):

    train_para, dev_para, test_para = {}, {}, {}
    csv_data = pd.read_csv(filename)

    for _, row in csv_data.iterrows():

        para_id = int(row['Paragraph ID'])
        para_data = paragraph_result[para_id]
        partition = row['Partition']
        if partition == 'train':
            train_para[para_id] = para_data
        elif partition == 'dev':
            dev_para[para_id] = para_data
        elif partition == 'test':
            test_para[para_id] = para_data
        
    print('Number of train paragraphs: ', len(train_para))
    print('Number of dev paragraphs: ', len(dev_para))
    print('Number of test paragraphs: ', len(test_para))

    return train_para, dev_para, test_para


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-para_file', type=str, default='data/Paragraphs.csv', help='path to the paragraph csv')
    parser.add_argument('-state_file', type=str, default='data/State_change_annotations.csv', 
                        help='path to the state annotation csv')
    parser.add_argument('-split_file', type=str, default='data/train_dev_test.csv', 
                        help='path to the csv that annotates the train/dev/test split')
    parser.add_argument('-log_dir', type=str, default='logs',
                        help='directory to store the intermediate outputs')
    parser.add_argument('-store_dir', type=str, default='data/', 
                        help='directory that you would like to store the generated instances')
    opt = parser.parse_args()

    print('Received arguments:')
    print(opt)
    print('-' * 50)

    paragraph_result = read_paragraph(opt.para_file)
    train_para, dev_para, test_para = read_split(opt.split_file, paragraph_result)

    log_file = open(f'{opt.log_dir}/info.log', 'w', encoding='utf-8')
    # save the instances to JSON files
    print('Dev Set......')
    dev_instances = read_annotation(opt.state_file, dev_para, log_file, test = False)
    json.dump(dev_instances, open(os.path.join(opt.store_dir, 'dev.json'), 'w', encoding='utf-8'),
                ensure_ascii=False, indent=4)
                
    print('Testing Set......')
    test_instances = read_annotation(opt.state_file, test_para, log_file, test = True)
    json.dump(test_instances, open(os.path.join(opt.store_dir, 'test.json'), 'w', encoding='utf-8'),
                ensure_ascii=False, indent=4)

    print('Training Set......')
    train_instances = read_annotation(opt.state_file, train_para, log_file, test = False)
    json.dump(train_instances, open(os.path.join(opt.store_dir, 'train.json'), 'w', encoding='utf-8'),
                ensure_ascii=False, indent=4)

    print('[INFO] JSON files saved successfully.')

    log_file.close()
