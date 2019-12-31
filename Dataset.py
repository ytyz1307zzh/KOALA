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


class ProparaDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str, is_test: bool):
        super(ProparaDataset, self).__init__()

        print('[INFO] Starting load...')
        print(f'[INFO] Load data from {data_path}')
        start_time = time.time()

        self.dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        self.state2idx = state2idx
        self.idx2state = idx2state
        self.is_test = is_test

        print(f'[INFO] {len(self.dataset)} instances of data loaded. Time Elapse: {time.time() - start_time}s')

    
    def __len__(self):
        return len(self.dataset)


    def get_mask(self, mention_idx: List[int], para_len: int) -> List[int]:
        """
        Given a list of mention positions of the entity/verb/location in a paragraph,
        compute the mask of it.
        """
        return [1 if i in mention_idx else 0 for i in range(para_len)]


    def __getitem__(self, index: int):

        instance = self.dataset[index]

        entity_name = instance['entity']  # used in the evaluation process
        para_id = instance['id']  # used in the evaluation process
        total_tokens = instance['total_tokens']  # used in compute mask vector
        total_sents = instance['total_sents']
        total_loc_cands = instance['total_loc_candidates']
        loc_cand_list = instance['loc_cand_list']

        metadata = {'para_id': para_id,
                    'entity': entity_name,
                    'total_sents': total_sents,
                    'total_loc_cands': total_loc_cands,
                    'loc_cand_list': loc_cand_list,
                    'raw_gold_loc': instance['gold_loc_seq']
                    }
        paragraph = instance['paragraph'].strip().split()  # Elmo processes list of words     
        assert len(paragraph) == total_tokens
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

        # (num_sent, num_tokens)
        entity_mask_list = torch.IntTensor([self.get_mask(sent['entity_mention'], total_tokens) for sent in sentence_list])
        # (num_sent, num_tokens)
        verb_mask_list = torch.IntTensor([self.get_mask(sent['verb_mention'], total_tokens) for sent in sentence_list])
        # (num_cand, num_sent, num_tokens)
        loc_mask_list = torch.IntTensor([[self.get_mask(sent['loc_mention_list'][idx], total_tokens) for sent in sentence_list]
                                            for idx in range(total_loc_cands)])

        sample = {'metadata': metadata,
                  'paragraph': paragraph,
                  'sentences': sentences,
                  'gold_loc_seq': gold_loc_seq,
                  'gold_state_seq': gold_state_seq,
                  'entity_mask': entity_mask_list,
                  'verb_mask': verb_mask_list,
                  'loc_mask': loc_mask_list
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
        max_tokens = max([len(inst['paragraph']) for inst in batch])
        max_cands = max([inst['metadata']['total_loc_cands'] for inst in batch])
        batch_size = len(batch)

        # pad according to max_len
        batch = list(map(lambda x: self.pad_instance(x, max_sents = max_sents, 
                                                        max_tokens = max_tokens, 
                                                        max_cands = max_cands), batch))

        metadata = list(map(lambda x: x['metadata'], batch))
        paragraph = list(map(lambda x: x['paragraph'], batch))
        sentences = list(map(lambda x: x['sentences'], batch))
        gold_loc_seq = torch.stack(list(map(lambda x: x['gold_loc_seq'], batch)))
        gold_state_seq = torch.stack(list(map(lambda x: x['gold_state_seq'], batch)))
        entity_mask = torch.stack(list(map(lambda x: x['entity_mask'], batch)))
        verb_mask = torch.stack(list(map(lambda x: x['verb_mask'], batch)))
        loc_mask = torch.stack(list(map(lambda x: x['loc_mask'], batch)))

        # check the dimension of the data
        assert len(metadata) == len(paragraph) == batch_size
        assert gold_loc_seq.size() == gold_state_seq.size() == (batch_size, max_sents)
        assert entity_mask.size() == verb_mask.size() == (batch_size, max_sents, max_tokens)
        assert loc_mask.size() == (batch_size, max_cands, max_sents, max_tokens)

        return {'metadata': metadata,
                'paragraph': paragraph,  # unpadded, 2-dimension
                'sentences': sentences,  # unpadded, 2-dimension
                'gold_loc_seq': gold_loc_seq,
                'gold_state_seq': gold_state_seq,
                'entity_mask': entity_mask,
                'verb_mask': verb_mask,
                'loc_mask': loc_mask
                }

    
    def pad_instance(self, instance: Dict, max_sents: int, max_tokens: int, max_cands: int) -> Dict:
        """
        Pad the data fields of a certain instance.

        args: 
            instance - instance to pad
            max_sents -  maximum number of sentences in this batch
            max_tokens -  maximum number of tokens in this batch
        """
        instance['gold_state_seq'] = self.pad_tensor(instance['gold_state_seq'], pad = max_sents, dim = 0, pad_val = PAD_STATE)
        instance['gold_loc_seq'] = self.pad_tensor(instance['gold_loc_seq'], pad = max_sents, dim = 0, pad_val = PAD_LOC)
        
        instance['entity_mask'] = self.pad_mask_list(instance['entity_mask'], max_sents = max_sents, max_tokens = max_tokens)
        instance['verb_mask'] = self.pad_mask_list(instance['verb_mask'], max_sents = max_sents, max_tokens = max_tokens)
        instance['loc_mask'] = self.pad_mask_list(instance['loc_mask'], max_sents = max_sents,
                                                     max_tokens = max_tokens, max_cands = max_cands)

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

        
