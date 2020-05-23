'''
 @Date  : 12/09/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import json
import torch
from typing import List, Set, Dict
import numpy as np
from Constants import *
import re
from spacy.lang.en import STOP_WORDS
STOP_WORDS = set(STOP_WORDS) - {'bottom', 'serious', 'top', 'alone', 'around', 'used', 'behind', 'side', 'mine', 'well'}
from ConceptNet.Stemmer import PorterStemmer
stemmer = PorterStemmer()

max_num_candidates = 0
max_num_tokens = 0
max_num_sents = 0

def count_maximum():
    """
    Count the max number of location candidates / tokens / sentences in the dataset.
    This is just an auxiliary function and is not actually used in the model,
    because all instances are padded to the longest sequence in the certain batch, rather than the whole dataset.
    """
    def load_data(json_file):
        global max_num_candidates
        global max_num_tokens
        global max_num_sents
        data = json.load(open(json_file, 'r', encoding='utf-8'))
        print('number of instances: ', len(data))
        for instance in data:
            num_candidates = instance['total_loc_candidates']
            num_tokens = instance['total_tokens']
            num_sents = instance['total_sents']
            if num_candidates > max_num_candidates:
                max_num_candidates = num_candidates
            if num_tokens > max_num_tokens:
                max_num_tokens = num_tokens
            if num_sents > max_num_sents:
                max_num_sents = num_sents

    load_data('./data/train.json')
    print('train')
    load_data('./data/dev.json')
    print('dev')
    load_data('./data/test.json')
    print('test')

    print(f'max number of candidates: {max_num_candidates}')
    print(f'max number of tokens: {max_num_tokens}')
    print(f'max number of sentences: {max_num_sents}')


def count_average():
    """
    Count the average number of location candidates / tokens / sentences in the dataset.
    This is just an auxiliary function for statistics and is not actually used in the model.
    """
    def load_data(json_file):
        data = json.load(open(json_file, 'r', encoding='utf-8'))
        num_instances = len(data)
        print('number of instances: ', num_instances)
        total_candidates = 0
        total_tokens = 0
        total_sents = 0

        for instance in data:
            num_candidates = instance['total_loc_candidates']
            num_tokens = instance['total_tokens']
            num_sents = instance['total_sents']
            total_candidates += num_candidates
            total_tokens += num_tokens
            total_sents += num_sents

        print(f'Average number of location candidates: {total_candidates / num_instances}')
        print(f'Average number of tokens: {total_tokens / num_instances}')
        print(f'Average number of sentences: {total_sents / num_instances}')

    print('train')
    load_data('./data/train.json')
    print('dev')
    load_data('./data/dev.json')
    print('test')
    load_data('./data/test.json')


def prepare_input_text_pair(tokenizer, paragraphs, wiki):
    """
    Prepare input for the model. Format: <CLS> paragraph <SEP> wiki <SEP>
    Truncate wiki if overflow.
    Args:
        paragraphs - (batch,) each element is a string
        wiki - (batch, num_wiki) each element is a string
    Return:
        batch_input - (batch * num_wiki, max_tokens)
    """
    batch_text_pair = []
    assert len(paragraphs) == len(wiki)
    batch_size = len(paragraphs)
    max_wiki = len(wiki[0])

    for i in range(batch_size):
        text_pair = [(paragraphs[i], wiki[i][j]) for j in range(max_wiki)]
        batch_text_pair.extend(text_pair)

    outputs = tokenizer.batch_encode_plus(batch_text_pair, add_special_tokens=True, max_length=tokenizer.max_len,
                                          truncation_strategy='only_second', return_tensors='pt',
                                          return_token_type_ids=True)

    input_ids = outputs['input_ids']
    token_type_ids = outputs['token_type_ids']

    return input_ids, token_type_ids


def find_allzero_rows(vector: torch.IntTensor) -> torch.BoolTensor:
    """
    Find all-zero rows of a given tensor, which is of size (batch, max_sents, max_tokens).
    This function is used to find unmentioned sentences of a certain entity/location.
    So the input tensor is typically a entity_mask or loc_mask.
    Return:
        a BoolTensor indicating that a all-zero row is True. Convenient for masked_fill.
    """
    assert vector.dtype == torch.int
    column_sum = torch.sum(vector, dim = -1)
    return column_sum == 0


def compute_state_accuracy(pred: List[List[int]], gold: List[List[int]], pad_value: int) -> (int, int):
    """
    Given the predicted tags and gold tags, compute the prediction accuracy.
    Note that we first need to deal with the padded parts of the gold tags.
    """
    assert len(pred) == len(gold)
    unpad_gold = [unpad(li, pad_value = pad_value) for li in gold]
    correct_pred = 0
    total_pred = 0

    for i in range(len(pred)):
        assert len(pred[i]) == len(unpad_gold[i])
        total_pred += len(pred[i])
        correct_pred += np.sum(np.equal(pred[i], unpad_gold[i]))

    return correct_pred.item(), total_pred


def compute_loc_accuracy(logits: torch.FloatTensor, gold: torch.IntTensor, pad_value: int) -> (int, int):
    """
    Given the generated location logits and the gold location sequence, compute the location prediction accuracy.
    Args:
        logits - size (batch, max_sents, max_cands)
        gold - size (batch, max_sents)
        pad_value - elements with this value will not count in accuracy
    """
    pred = torch.argmax(logits, dim = -1)
    assert pred.size() == gold.size()

    total_pred = torch.sum(gold != pad_value)  # total number of valid elements
    correct_pred = torch.sum(pred == gold)  # the model cannot predict PAD, NIL or UNK, so all padded positions should be false

    return correct_pred.item(), total_pred.item()


def get_pred_loc(loc_logits: torch.Tensor, gold_loc_seq: torch.IntTensor) -> List[List[int]]:
    """
    Get the predicted location sequence from raw logits.
    Note that loc_logits should be MASKED while gold_loc_seq should NOT.
    Args:
        loc_logits - raw logits, with padding elements set to -inf (masked). (batch, max_sents, max_cands)
        gold_loc_seq - gold location sequence without masking. (batch, max_sents)
    """
    assert gold_loc_seq.size() == (loc_logits.size(0), loc_logits.size(1))
    argmax_loc = torch.argmax(loc_logits, dim = -1)
    assert argmax_loc.size() == gold_loc_seq.size()
    argmax_loc = argmax_loc.masked_fill(mask = (gold_loc_seq == PAD_LOC), value = PAD_LOC).tolist()

    pred_loc = []
    for inst in argmax_loc:
        pred_loc.append([x for x in inst if x != PAD_LOC])

    return pred_loc


def get_report_time(total_batches: int, report_times: int, grad_accum_step: int) -> List[int]:
    """
    Given the total number of batches in an epoch and the report times per epoch,
    compute on which timesteps do we need to report
    e.g. total_batches = 25, report_times = 3, then we should report on batch number [8, 16, 25]
    Batch numbers start from one.
    """
    report_span = round(total_batches / grad_accum_step / report_times)
    report_batch = [i * report_span * grad_accum_step for i in range(1, report_times)]
    report_batch.append(total_batches // grad_accum_step * grad_accum_step)
    return report_batch


def bert_subword_map(origin_tokens: List[str], tokens: List[str]) -> List[int]:
    """
    Map the original tokens to tokenized BERT sub-tokens.
    Args:
        origin_tokens: List of original tokens.
        tokens: List of sub-tokens given by a BertTokenizer.
    Return:
        offset: A list of original token ids, each indexed by a sub-token id.
    """
    offset = []
    j = 0
    cur_token = ''

    for i in range(len(tokens)):
        if cur_token != '':
            cur_token += tokens[i]
            if '##' in cur_token:
                cur_token = cur_token.replace('##', '')
        else:
            cur_token = tokens[i]
        if cur_token != origin_tokens[j].lower() and cur_token != '[UNK]':
            offset.append(j)
        else:
            offset.append(j)
            j += 1
            cur_token = ''

    return offset


def remove_stopword(paragraph: str) -> Set[str]:
    """
    Acquire all content words from a paragraph.
    """
    paragraph = paragraph.lower().strip().split()
    return {word for word in paragraph if word not in STOP_WORDS and word.isalpha()}


def find_relevant_triple(gold_loc_seq: List[str], gold_state_seq: List[str], verb_dict: Dict, cpnet_triples: List[str]):
    """
    Find relevant triples for computing attention loss.
    Two categories: state-relevant and location-relevant
    """

    assert len(gold_state_seq) == len(gold_loc_seq) - 1
    state_rel_labels, loc_rel_labels = [], []

    # deal with location 0
    loc_rel_list = [0 for _ in range(len(cpnet_triples))]
    loc_rel_labels.append(loc_rel_list)

    for sent_idx in range(len(gold_state_seq)):
        state_rel_list, loc_rel_list = [], []
        location = gold_loc_seq[sent_idx + 1]
        state = gold_state_seq[sent_idx]

        state_token_set = None
        if state == 'C':
            state_token_set = set(verb_dict['create'])
        elif state == 'M':
            state_token_set = set(verb_dict['move'])
        elif state == 'D':
            state_token_set = set(verb_dict['destroy'])

        loc_token_set = remove_stopword(location)
        loc_token_set = set(map(stemmer.stem, loc_token_set))

        for triple in cpnet_triples:
            token_set = remove_stopword(triple)
            token_set = set(map(stemmer.stem, token_set))

            if state_token_set is not None and token_set.intersection(state_token_set):
                state_rel_list.append(1)
            else:
                state_rel_list.append(0)

            if state in ['C', 'M'] and token_set.intersection(loc_token_set):
                loc_rel_list.append(1)
            else:
                loc_rel_list.append(0)

        state_rel_labels.append(state_rel_list)
        loc_rel_labels.append(loc_rel_list)

    state_rel_labels = torch.tensor(state_rel_labels, dtype=torch.int)
    loc_rel_labels = torch.tensor(loc_rel_labels, dtype=torch.int)

    return state_rel_labels, loc_rel_labels


def unpad(source: List[int], pad_value: int) -> List[int]:
    """
    Remove padded elements from a list
    """
    return [x for x in source if x != pad_value]


def mean(source: List) -> float:
    return sum(source) / len(source)

