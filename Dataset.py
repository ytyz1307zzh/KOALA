'''
 @Date  : 12/09/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import torch
import json
import os
import time
import numpy as np
from typing import List, Dict
from Constants import *
from utils import *


class ProparaDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str, cpnet_path: str, tokenizer, is_test: bool):
        super(ProparaDataset, self).__init__()

        print('[INFO] Starting load...')
        print(f'[INFO] Load data from {data_path}')
        start_time = time.time()

        self.dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        self.tokenizer = tokenizer
        self.cpnet = self.read_cpnet(cpnet_path)
        self.state2idx = state2idx
        self.idx2state = idx2state
        self.is_test = is_test

        print(f'[INFO] {len(self.dataset)} instances of data loaded. Time Elapse: {time.time() - start_time}s')

    
    def __len__(self):
        return len(self.dataset)


    def read_cpnet(self, cpnet_path: str):
        cpnet = json.load(open(cpnet_path, 'r', encoding='utf-8'))
        cpnet_dict = {}

        for instance in cpnet:
            para_id = instance['id']
            entity = instance['entity']
            cpnet_dict[f'{para_id}-{entity}'] = instance

        return cpnet_dict


    def convert_wordmask_to_subwordmask(self, mask, offset_map):
        num_subword = len(offset_map)
        return [1 if mask[offset_map[i]] == 1 else 0 for i in range(num_subword)]


    def get_word_mask(self, mention_idx: List[int], offset_map: List[int], para_len: int) -> List[int]:
        """
        Given a list of mention positions of the entity/verb/location in a paragraph,
        compute the mask of it.
        """
        word_mask =  [1 if i in mention_idx else 0 for i in range(para_len)]
        subword_mask = self.convert_wordmask_to_subwordmask(mask=word_mask, offset_map=offset_map)
        # prepare positions for <CLS> & <SEP>
        subword_mask = [0] + subword_mask + [0]
        return subword_mask


    def get_sentence_mention(self, sentence_list: List, para_len: int):
        """
        Get the indexes of a given sentence.
        """
        sentence_mentions = []
        sentence_lengths = [x['total_tokens'] for x in sentence_list]
        assert sum(sentence_lengths) == para_len
        prev_tokens = 0

        for length in sentence_lengths:
            mention_idx = [idx for idx in range(prev_tokens, prev_tokens + length)]
            sentence_mentions.append(mention_idx)
            prev_tokens += length

        return sentence_mentions


    def find_cpnet(self, para_id: int, entity: str):
        cpnet_triples = self.cpnet[f'{para_id}-{entity}']['cpnet']
        cpnet_sentences = [triple.split(', ')[10] for triple in cpnet_triples]
        return cpnet_sentences


    def __getitem__(self, index: int):

        instance = self.dataset[index]

        entity_name = instance['entity']  # used in the evaluation process
        para_id = instance['id']  # used in the evaluation process
        total_words = instance['total_tokens']  # used in compute mask vector
        total_sents = instance['total_sents']
        total_loc_cands = instance['total_loc_candidates']
        loc_cand_list = instance['loc_cand_list']

        paragraph = instance['paragraph']
        assert len(paragraph.strip().split()) == total_words
        tokens = self.tokenizer.tokenize(paragraph)
        if isinstance(self.tokenizer, BertTokenizer):
            offset_map = bert_subword_map(origin_tokens=paragraph.strip().split(), tokens=tokens)
        else:
            raise ValueError(f'Did not provide mapping function for tokenizer {type(self.tokenizer)}')

        metadata = {'para_id': para_id,
                    'entity': entity_name,
                    'total_subwords': len(tokens)+2,  # subwords + <CLS> + <SEP>
                    'total_sents': total_sents,
                    'total_loc_cands': total_loc_cands,
                    'loc_cand_list': loc_cand_list,
                    'raw_gold_loc': instance['gold_loc_seq']
                    }

        gold_state_seq = torch.IntTensor([self.state2idx[label] for label in instance['gold_state_seq']])

        loc2idx = {loc_cand_list[idx]: idx for idx in range(total_loc_cands)}
        loc2idx['-'] = NIL_LOC
        loc2idx['?'] = UNK_LOC
        # note that the loc_cand_list in exactly "idx2loc" (excluding '?' and '-')

        # for train and dev sets, all gold locations should have been included in candidate set
        # for test set, the gold location may not in the candidate set
        gold_loc_seq = torch.IntTensor([loc2idx[loc] if (loc in loc_cand_list or loc in ['-', '?']) else UNK_LOC
                                            for loc in instance['gold_loc_seq'][1:]])

        assert gold_loc_seq.size() == gold_state_seq.size()
        sentence_list = instance['sentence_list']
        sentences = [x['sentence'] for x in sentence_list]
        assert total_sents == len(sentence_list)
        sentence_mention = self.get_sentence_mention(sentence_list, total_words)

        # (num_sent, num_tokens)
        sentence_mask_list = torch.IntTensor([self.get_word_mask(sentence_mention[i], offset_map, total_words)
                                            for i in range(len(sentence_mention))])
        # (num_sent, num_tokens)
        entity_mask_list = torch.IntTensor([self.get_word_mask(sent['entity_mention'], offset_map, total_words)
                                            for sent in sentence_list])
        # (num_sent, num_tokens)
        verb_mask_list = torch.IntTensor([self.get_word_mask(sent['verb_mention'], offset_map, total_words)
                                          for sent in sentence_list])
        # (num_cand, num_sent, num_tokens)
        loc_mask_list = torch.IntTensor([[self.get_word_mask(sent['loc_mention_list'][idx], offset_map, total_words)
                                          for sent in sentence_list] for idx in range(total_loc_cands)])
        
        # map word positions to sub-word positions

        cpnet_triples = self.find_cpnet(para_id=para_id, entity=entity_name)

        sample = {'metadata': metadata,
                  'paragraph': paragraph,
                  'sentences': sentences,
                  'gold_loc_seq': gold_loc_seq,
                  'gold_state_seq': gold_state_seq,
                  'sentence_mask': sentence_mask_list,
                  'entity_mask': entity_mask_list,
                  'verb_mask': verb_mask_list,
                  'loc_mask': loc_mask_list,
                  'cpnet': cpnet_triples
                }

        return sample


# For paragraphs, we pad them to the max number of tokens in a batch
# For sentences, we pad them to the max number of sentences in a batch
# For location candidates, we pad them to the max number of location candidates in a batch
class Collate:
    """
    A variant of callate_fn that pads according to the longest sequence in
    a batch of sequences, turn List[Dict] -> Dict[List]
    """
    def __init__(self):
        pass


    def __call__(self, batch):
        return self.collate(batch)


    def collate(self, batch: List[Dict]):
        """
        Convert a list of dict instances to a dict of batched tensors

        args:
            batch - list of instances constructed by dataset

        reutrn:
            batch - a dict, contains lists of data fields
        """
        # find max number of sentences & tokens
        max_sents = max([inst['metadata']['total_sents'] for inst in batch])
        max_tokens = max([inst['metadata']['total_subwords'] for inst in batch])
        max_cands = max([inst['metadata']['total_loc_cands'] for inst in batch])
        max_cpnet = max([len(inst['cpnet']) for inst in batch])
        batch_size = len(batch)

        # pad according to max_len
        batch = list(map(lambda x: self.pad_instance(x, max_sents = max_sents, 
                                                        max_tokens = max_tokens, 
                                                        max_cands = max_cands,
                                                        max_cpnet = max_cpnet), batch))

        metadata = list(map(lambda x: x['metadata'], batch))
        paragraph = list(map(lambda x: x['paragraph'], batch))
        sentences = list(map(lambda x: x['sentences'], batch))
        cpnet = list(map(lambda x: x['cpnet'], batch))
        gold_loc_seq = torch.stack(list(map(lambda x: x['gold_loc_seq'], batch)))
        gold_state_seq = torch.stack(list(map(lambda x: x['gold_state_seq'], batch)))
        sentence_mask = torch.stack(list(map(lambda x: x['sentence_mask'], batch)))
        entity_mask = torch.stack(list(map(lambda x: x['entity_mask'], batch)))
        verb_mask = torch.stack(list(map(lambda x: x['verb_mask'], batch)))
        loc_mask = torch.stack(list(map(lambda x: x['loc_mask'], batch)))

        # check the dimension of the data
        assert len(metadata) == len(paragraph) == len(sentences) == len(cpnet) == batch_size
        assert gold_loc_seq.size() == gold_state_seq.size() == (batch_size, max_sents)
        assert sentence_mask.size() == entity_mask.size() == verb_mask.size() == (batch_size, max_sents, max_tokens)
        assert loc_mask.size() == (batch_size, max_cands, max_sents, max_tokens)

        return {'metadata': metadata,
                'paragraph': paragraph,  # unpadded, 1-dimension
                'sentences': sentences,  # unpadded, 2-dimension
                'gold_loc_seq': gold_loc_seq,
                'gold_state_seq': gold_state_seq,
                'sentence_mask': sentence_mask,
                'entity_mask': entity_mask,
                'verb_mask': verb_mask,
                'loc_mask': loc_mask,
                'cpnet': cpnet
                }

    def pad_instance(self, instance: Dict, max_sents: int, max_tokens: int,
                     max_cands: int, max_cpnet: int) -> Dict:
        """
        Pad the data fields of a certain instance.

        args: 
            instance - instance to pad
            max_sents -  maximum number of sentences in this batch
            max_tokens -  maximum number of tokens in this batch
        """
        instance['gold_state_seq'] = self.pad_tensor(instance['gold_state_seq'], pad = max_sents, dim = 0, pad_val = PAD_STATE)
        instance['gold_loc_seq'] = self.pad_tensor(instance['gold_loc_seq'], pad = max_sents, dim = 0, pad_val = PAD_LOC)

        instance['sentence_mask'] = self.pad_mask_list(instance['sentence_mask'], max_sents = max_sents, max_tokens = max_tokens)
        instance['entity_mask'] = self.pad_mask_list(instance['entity_mask'], max_sents = max_sents, max_tokens = max_tokens)
        instance['verb_mask'] = self.pad_mask_list(instance['verb_mask'], max_sents = max_sents, max_tokens = max_tokens)
        instance['loc_mask'] = self.pad_mask_list(instance['loc_mask'], max_sents = max_sents,
                                                     max_tokens = max_tokens, max_cands = max_cands)

        instance['cpnet'] = self.pad_cpnet(instance['cpnet'], max_num = max_cpnet)

        return instance

    
    def pad_mask_list(self, vec: torch.Tensor, max_sents: int, max_tokens: int, max_cands: int = None) -> torch.Tensor:
        """
        Pad a tensor of mask list
        """
        tmp_vec = self.pad_tensor(vec, pad = max_tokens, dim = -1)
        tmp_vec = self.pad_tensor(tmp_vec, pad = max_sents, dim = -2)
        if max_cands is not None:
            tmp_vec = self.pad_tensor(tmp_vec, pad = max_cands, dim = -3)

        return tmp_vec

    
    def pad_tensor(self, vec: torch.Tensor, pad: int, dim: int, pad_val: int = 0) -> torch.Tensor:
        """
        Pad a tensor on a given dimension to a given size.

        args:
            vec - tensor to pad
            pad - the size to pad to
            dim - dimension to pad

        return:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.size())
        pad_size[dim] = pad - vec.size(dim)
        pad_vec = torch.zeros(*pad_size, dtype = vec.dtype)

        if pad_val != 0:
            pad_vec.fill_(pad_val)

        return torch.cat([vec, pad_vec], dim = dim)


    def pad_cpnet(self, data: List[str], max_num: int):
        """
        Pad the cpnet triples to max number.
        """
        return data + ['' for _ in range(max_num - len(data))]

        
