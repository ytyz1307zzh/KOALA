'''
 @Date  : 12/11/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os
import time
import numpy as np
from typing import List, Dict
from Constants import *
import itertools
from utils import *
from torchcrf import CRF
import argparse
import pdb


class NCETModel(nn.Module):

    def __init__(self, opt: argparse.Namespace, is_test: bool):

        super(NCETModel, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.embed_size = MODEL_HIDDEN[opt.plm_model_name]

        self.TokenEncoder = nn.LSTM(input_size = self.embed_size, hidden_size = opt.hidden_size,
                                    num_layers = 1, batch_first = True, bidirectional = True)
        self.Dropout = nn.Dropout(p = opt.dropout)

        # fixed pretrained language model
        assert opt.plm_model_name in MODEL_HIDDEN.keys(), 'Wrong model name provided'
        plm_model_class, plm_tokenizer_class, plm_config_class = MODEL_CLASSES[opt.plm_model_class]

        self.plm_config = plm_config_class.from_pretrained(opt.plm_model_name)
        self.plm_tokenizer = plm_tokenizer_class.from_pretrained(opt.plm_model_name)

        # ConceptNet encoder
        if is_test:
            self.cpnet_encoder = plm_model_class(config=self.plm_config)  # use saved parameters
        else:
            self.cpnet_encoder = plm_model_class.from_pretrained(opt.plm_model_name)
        print(f'[INFO] Loaded {opt.plm_model_name} for ConceptNet encoder')
        for param in self.cpnet_encoder.parameters():
            param.requires_grad = False

        self.CpnetEncoder = FixedSentEncoder(opt)
        if not opt.no_wiki and not opt.embed_plm_path:
            self.EmbeddingLayer = TokenEmbedding(opt, plm_config=self.plm_config,
                                                 plm_model_class=plm_model_class,
                                                 pad_token_id=self.plm_tokenizer.pad_token_id,
                                                 is_test=is_test)
        elif not opt.no_wiki:
            if is_test:
                self.embed_encoder = plm_model_class(config=self.plm_config)  # use saved parameters
            else:
                self.embed_encoder = plm_model_class.from_pretrained(opt.embed_plm_path)
            print(f'[INFO] Loaded {opt.embed_plm_path} for embedding language model')
            if not opt.finetune:
                for param in self.embed_encoder.parameters():
                    param.requires_grad = False

        # state tracking modules
        self.StateTracker = StateTracker(opt)
        self.CRFLayer = CRF(NUM_STATES, batch_first = True)

        # location prediction modules
        self.LocationPredictor = LocationPredictor(opt)
        self.CrossEntropy = nn.CrossEntropyLoss(ignore_index = PAD_LOC, reduction = 'mean')
        self.BinaryCrossEntropy = nn.BCELoss(reduction='mean')

        self.is_test = is_test


    def forward(self, token_ids: torch.Tensor, token_type_ids: torch.Tensor, entity_mask: torch.IntTensor,
                verb_mask: torch.IntTensor, loc_mask: torch.IntTensor, gold_loc_seq: torch.IntTensor,
                gold_state_seq: torch.IntTensor,num_cands: torch.IntTensor, sentence_mask: torch.IntTensor,
                cpnet_triples: List, state_rel_labels: torch.IntTensor, loc_rel_labels: torch.IntTensor, print_hidden):
        """
        Args:
            token_ids: size (batch * max_wiki, max_ctx_tokens)
            token_type_ids: size (batch * max_wiki, max_ctx_tokens)
            *_mask: size (batch, max_sents, max_tokens)
            gold_loc_seq: size (batch, max_sents)
            gold_state_seq: size (batch, max_sents)
            state_rel_labels: size (batch, max_sents, max_cpnet)
            loc_rel_labels: size (batch, max_sents, max_cpnet)
            num_cands: size (batch,)
        """
        assert entity_mask.size(-2) == verb_mask.size(-2) == loc_mask.size(-2) == gold_state_seq.size(-1) == gold_loc_seq.size(-1)
        assert entity_mask.size(-1) == verb_mask.size(-1) == loc_mask.size(-1)
        batch_size = entity_mask.size(0)
        max_tokens = entity_mask.size(-1)
        max_sents = gold_state_seq.size(-1)
        max_cands = loc_mask.size(-3)

        if self.opt.no_wiki:
            attention_mask = (token_ids != self.plm_tokenizer.pad_token_id).to(torch.int)
            plm_outputs = self.cpnet_encoder(token_ids, attention_mask=attention_mask)
            embeddings = plm_outputs[0]  # hidden states at the last layer, (batch, max_tokens, plm_hidden_size)
        elif not self.opt.no_wiki and self.opt.embed_plm_path:
            attention_mask = (token_ids != self.plm_tokenizer.pad_token_id).to(torch.int)
            plm_outputs = self.embed_encoder(token_ids, attention_mask=attention_mask)
            embeddings = plm_outputs[0]  # hidden states at the last layer, (batch, max_tokens, plm_hidden_size)
        else:
            assert token_ids.size(-1) == token_type_ids.size(-1)
            assert token_ids.size(0) % batch_size == 0
            max_wiki = token_ids.size(0) // batch_size
            embeddings = self.EmbeddingLayer(token_ids=token_ids, token_type_ids=token_type_ids, num_wiki=max_wiki)

        token_rep, _ = self.TokenEncoder(embeddings)  # (batch, max_tokens, 2*hidden_size)
        token_rep = self.Dropout(token_rep)
        assert token_rep.size() == (batch_size, max_tokens, 2 * self.hidden_size)

        cpnet_rep = self.CpnetEncoder(cpnet_triples, tokenizer=self.plm_tokenizer, encoder=self.cpnet_encoder)

        # state change prediction
        # size (batch, max_sents, NUM_STATES)
        tag_logits, state_attn_probs = self.StateTracker(encoder_out = token_rep, entity_mask = entity_mask,
                                                         verb_mask = verb_mask, sentence_mask = sentence_mask,
                                                         cpnet_triples = cpnet_triples, cpnet_rep = cpnet_rep)
        tag_mask = (gold_state_seq != PAD_STATE) # mask the padded part so they won't count in loss
        log_likelihood = self.CRFLayer(emissions = tag_logits, tags = gold_state_seq.long(), mask = tag_mask, reduction = 'token_mean')

        state_loss = -log_likelihood  # State classification loss is negative log likelihood
        pred_state_seq = self.CRFLayer.decode(emissions=tag_logits, mask=tag_mask)
        assert len(pred_state_seq) == batch_size
        correct_state_pred, total_state_pred = compute_state_accuracy(pred=pred_state_seq, gold=gold_state_seq.tolist(),
                                                        pad_value=PAD_STATE)

        # location prediction
        # size (batch, max_cands, max_sents)
        loc_logits, loc_attn_probs = self.LocationPredictor(encoder_out = token_rep, entity_mask = entity_mask,
                                                            loc_mask = loc_mask, sentence_mask = sentence_mask,
                                                            cpnet_triples = cpnet_triples, cpnet_rep = cpnet_rep)
        loc_logits = loc_logits.transpose(-1, -2)  # size (batch, max_sents, max_cands)
        masked_loc_logits = self.mask_loc_logits(loc_logits = loc_logits, num_cands = num_cands)  # (batch, max_sents, max_cands)
        masked_gold_loc_seq = self.mask_undefined_loc(gold_loc_seq = gold_loc_seq, mask_value = PAD_LOC)  # (batch, max_sents)
        loc_loss = self.CrossEntropy(input = masked_loc_logits.view(batch_size * max_sents, max_cands),
                                     target = masked_gold_loc_seq.view(batch_size * max_sents).long())
        correct_loc_pred, total_loc_pred = compute_loc_accuracy(logits = masked_loc_logits, gold = masked_gold_loc_seq,
                                                                pad_value = PAD_LOC)

        gold_loc_mask = self.get_gold_loc_mask(loc_mask, gold_loc_seq)
        if loc_attn_probs is not None:
            loc_attn_probs = self.get_gold_attn_probs(loc_attn_probs, gold_loc_seq)
        attn_loss, total_attn_pred = self.get_attn_loss(state_attn_probs, loc_attn_probs, state_rel_labels, loc_rel_labels,
                                                        entity_mask, gold_loc_mask, print_hidden)

        if self.is_test:  # inference
            pred_loc_seq = get_pred_loc(loc_logits = masked_loc_logits, gold_loc_seq = gold_loc_seq)
            return pred_state_seq, pred_loc_seq, correct_state_pred, total_state_pred, correct_loc_pred, total_loc_pred

        return state_loss, loc_loss, attn_loss, correct_state_pred, total_state_pred, \
               correct_loc_pred, total_loc_pred, total_attn_pred


    def get_attn_loss(self, state_attn_probs, loc_attn_probs, state_rel_labels, loc_rel_labels,
                      entity_mask, gold_loc_mask, print_hidden):
        """
        Compute attention loss.
        All inputs: (batch, max_sents, max_cpnet)
        *_mask: (batch, max_sents, max_tokens)
        """
        # discard thos timesteps that are masked
        # state_sent_mask = torch.sum(entity_mask, dim=-1, keepdim=True)
        # loc_sent_mask = torch.sum(entity_mask, dim=-1, keepdim=True) + torch.sum(gold_loc_mask, dim=-1, keepdim=True)
        # state_rel_labels = state_rel_labels.masked_fill(mask=(state_sent_mask == 0), value=0)
        # loc_rel_labels = loc_rel_labels.masked_fill(mask=(loc_sent_mask == 0), value=0)

        pos_attn_probs = None
        if state_attn_probs is not None:
            state_pos_attn_probs = state_attn_probs.masked_select(mask=state_rel_labels.to(torch.bool))  # 1-D tensor
        if loc_attn_probs is not None:
            loc_pos_attn_probs = loc_attn_probs.masked_select(mask=loc_rel_labels.to(torch.bool))  # 1-D tensor

        if state_attn_probs is not None and loc_attn_probs is not None:
            pos_attn_probs = torch.cat([state_pos_attn_probs, loc_pos_attn_probs], dim=0)
        elif state_attn_probs is not None:
            pos_attn_probs = state_pos_attn_probs
        elif loc_attn_probs is not None:
            pos_attn_probs = loc_pos_attn_probs

        attn_loss = None
        total_attn_pred = None
        if pos_attn_probs is not None:
            gold_labels = torch.ones_like(pos_attn_probs)
            attn_loss = self.BinaryCrossEntropy(input=pos_attn_probs, target=gold_labels)
            total_attn_pred = gold_labels.size(0)

        return attn_loss, total_attn_pred


    def get_gold_attn_probs(self, loc_attn_probs, gold_loc_seq):
        batch_size = gold_loc_seq.size(0)
        max_sents = gold_loc_seq.size(1)
        max_cpnet = loc_attn_probs.size(-1)
        pick_loc_seq = gold_loc_seq.masked_fill(mask=(gold_loc_seq < 0), value=0)
        gold_attn_probs = []

        for i in range(batch_size):
            for j in range(max_sents):
                gold_attn_probs.append(loc_attn_probs[i][pick_loc_seq[i][j]][j])

        gold_attn_probs = torch.stack(gold_attn_probs, dim=0)
        return gold_attn_probs.view(batch_size, max_sents, max_cpnet)


    def get_gold_loc_mask(self, loc_mask, gold_loc_seq):
        batch_size = gold_loc_seq.size(0)
        max_sents = gold_loc_seq.size(1)
        max_tokens = loc_mask.size(-1)
        pick_loc_seq = gold_loc_seq.masked_fill(mask=(gold_loc_seq < 0), value=0)
        gold_loc_mask = []

        for i in range(batch_size):
            for j in range(max_sents):
                gold_loc_mask.append(loc_mask[i][pick_loc_seq[i][j]][j])

        gold_loc_mask = torch.stack(gold_loc_mask, dim=0)
        return gold_loc_mask.view(batch_size, max_sents, max_tokens)


    def mask_loc_logits(self, loc_logits, num_cands: torch.IntTensor):
        """
        Mask the padded candidates with an -inf score, so they will have a likelihood = 0 after softmax
        Args:
            loc_logits - output scores for each candidate in each sentence, size (batch, max_sents, max_cands)
            num_cands - total number of candidates in each instance of the given batch, size (batch,)
        """
        assert torch.max(num_cands) == loc_logits.size(-1)
        assert loc_logits.size(0) == num_cands.size(0)
        batch_size = loc_logits.size(0)
        max_cands = loc_logits.size(-1)

        # first, we create a mask tensor that masked all positions above the num_cands limit
        range_tensor = torch.arange(start = 1, end = max_cands + 1)
        if not self.opt.no_cuda:
            range_tensor = range_tensor.cuda()
        range_tensor = range_tensor.unsqueeze(dim = 0).expand(batch_size, max_cands)
        bool_range = torch.gt(range_tensor, num_cands.unsqueeze(dim = -1))  # find the off-limit positions
        assert bool_range.size() == (batch_size, max_cands)

        bool_range = bool_range.unsqueeze(dim = -2).expand_as(loc_logits)  # use this bool tensor to mask loc_logits
        masked_loc_logits = loc_logits.masked_fill(bool_range, value = float('-inf'))  # mask padded positions to -inf
        assert masked_loc_logits.size() == loc_logits.size()

        return masked_loc_logits


    def mask_undefined_loc(self, gold_loc_seq, mask_value: int):
        """
        Mask all undefined locations (NIL, UNK, PAD) in order not to count them in loss nor accuracy.
        Since these three special labels are all negetive, any position with a negative target label will be masked to mask_value.
        Args:
            gold_loc_seq - sequence of gold locations, size (batch, max_sents)
            mask_value - Should be the same label with ignore_index argument in cross-entropy.
        """
        negative_labels = torch.lt(gold_loc_seq, 0)
        masked_gold_loc_seq = gold_loc_seq.masked_fill(mask = negative_labels, value = mask_value)
        return masked_gold_loc_seq


    @staticmethod
    def expand_dim_3d(vec: torch.Tensor, loc_cands: int):
        """
        Expand a 3-dim vector in the batch dimension (dimension 0)
        """
        assert len(vec.size()) == 3
        batch_size = vec.size(0)
        seq_len = vec.size(1)
        rep_size = vec.size(2)
        vec = vec.unsqueeze(1).repeat(1, loc_cands, 1, 1)
        vec = vec.view(batch_size * loc_cands, seq_len, rep_size)
        return vec

    @staticmethod
    def expand_dim_2d(vec: torch.Tensor, loc_cands: int):
        """
        Expand a 2-dim vector in the batch dimension (dimension 0)
        """
        assert len(vec.size()) == 2
        batch_size = vec.size(0)
        seq_len = vec.size(1)
        vec = vec.unsqueeze(1).repeat(1, loc_cands, 1)
        vec = vec.view(batch_size * loc_cands, seq_len)
        return vec


class TokenEmbedding(nn.Module):
    """
    Token embedding layer to generate contextualized token embeddings
    """
    def __init__(self, opt: argparse.Namespace, plm_config, plm_model_class, pad_token_id, is_test: bool):
        super(TokenEmbedding, self).__init__()
        self.plm_config = plm_config
        self.embed_size = MODEL_HIDDEN[opt.plm_model_name]
        self.pad_token_id = pad_token_id
        self.input_token_type = 0
        self.cuda = not opt.no_cuda

        # Embedding language model
        if is_test:
            self.embed_encoder = plm_model_class(config=self.plm_config)  # use saved parameters
        else:
            self.embed_encoder = plm_model_class.from_pretrained(opt.plm_model_name)
        print(f'[INFO] Loaded {opt.plm_model_name} for embedding language model')
        if not opt.finetune:
            for param in self.embed_encoder.parameters():
                param.requires_grad = False

        self.weight_fc = Linear(d_in=self.embed_size, d_out=1, dropout=0)
        self.Dropout = nn.Dropout(p=opt.dropout)


    def forward(self, token_ids: torch.Tensor, token_type_ids: torch.Tensor, num_wiki: int):
        """
        Args:
            token_ids - (batch * num_wiki, max_ctx_tokens)
            token_type_ids - (batch * num_wiki, max_ctx_tokens)
        """
        assert token_ids.size() == token_type_ids.size()
        assert token_ids.size(0) % num_wiki == 0
        batch_size = token_ids.size(0) // num_wiki
        max_ctx_tokens = token_ids.size(-1)

        token_id_batches = [token_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                            for batch_idx in range(token_ids.size(0) // batch_size + 1)]
        token_type_id_batches = [token_type_ids[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                                 for batch_idx in range(token_type_ids.size(0) // batch_size + 1)]

        num_ori_tokens = torch.sum((token_type_ids == self.input_token_type), dim=-1)
        max_ori_tokens = torch.max(num_ori_tokens).item()

        unpad_token_mask = (token_ids.view(batch_size, num_wiki, max_ctx_tokens) != 0)
        unpad_token_sum = torch.sum(unpad_token_mask, dim=-1)
        ori_token_sum = num_ori_tokens.view_as(unpad_token_sum)
        second_sent_tokens = unpad_token_sum - ori_token_sum
        wiki_mask = (second_sent_tokens == 1)

        batch_last_hidden, batch_cls_hidden = [], []

        for batch_token_ids, batch_token_type_ids\
                in zip(token_id_batches, token_type_id_batches):

            mini_batch_size = batch_token_ids.size(0)
            if mini_batch_size == 0:
                continue

            attention_mask = (batch_token_ids != self.pad_token_id).to(torch.int)

            if self.cuda:
                batch_token_ids = batch_token_ids.cuda()
                batch_token_type_ids = batch_token_type_ids.cuda()
                attention_mask = attention_mask.cuda()

            outputs = self.embed_encoder(batch_token_ids, attention_mask=attention_mask,
                                         token_type_ids=batch_token_type_ids)
            last_hidden = outputs[0]
            cls_hidden = outputs[1]
            assert last_hidden.size() == (mini_batch_size, max_ctx_tokens, self.embed_size)
            assert cls_hidden.size() == (mini_batch_size, self.embed_size)

            second_sent_mask = batch_token_type_ids.unsqueeze(-1)
            last_hidden = last_hidden.masked_fill((second_sent_mask == 1), value=0)

            first_sent_hidden = torch.narrow(last_hidden, dim=1, start=0, length=max_ori_tokens)
            batch_last_hidden.append(first_sent_hidden)
            batch_cls_hidden.append(cls_hidden)

        batch_last_hidden = torch.stack(batch_last_hidden, dim=0)
        batch_cls_hidden = torch.stack(batch_cls_hidden, dim=0)
        batch_last_hidden = batch_last_hidden.view(batch_size, num_wiki, max_ori_tokens * self.embed_size)
        batch_cls_hidden = batch_cls_hidden.view(batch_size, num_wiki, self.embed_size)
        batch_last_hidden = self.Dropout(batch_last_hidden)
        batch_cls_hidden = self.Dropout(batch_cls_hidden)

        attn_weights = self.weight_fc(batch_cls_hidden).squeeze()  # (batch, num_wiki)
        attn_weights = attn_weights.masked_fill(wiki_mask, value=float('-inf'))
        attn_probs = F.softmax(attn_weights, dim=-1)  # (batch, num_wiki)

        is_nan = torch.isnan(attn_probs)
        attn_probs = attn_probs.masked_fill(is_nan, value = 1 / num_wiki)

        embedding = torch.bmm(attn_probs.unsqueeze(1), batch_last_hidden)  # (batch, 1, max_ctx_tokens * embed_size)
        embedding = embedding.squeeze().view(batch_size, max_ori_tokens, self.embed_size)

        return embedding


class StateTracker(nn.Module):
    """
    State tracking decoder: sentence-level Bi-LSTM + linear + CRF
    """
    def __init__(self, opt: argparse.Namespace):

        super(StateTracker, self).__init__()
        self.hidden_size = opt.hidden_size
        self.Decoder = nn.LSTM(input_size = 4 * opt.hidden_size, hidden_size = opt.hidden_size,
                                    num_layers = 1, batch_first = True, bidirectional = True)
        self.Dropout = nn.Dropout(p = opt.dropout)
        self.Hidden2Tag = Linear(d_in = 2 * opt.hidden_size, d_out = NUM_STATES, dropout = 0)
        self.CpnetMemory = CpnetMemory(opt, query_size = 4 * opt.hidden_size, input_size = 4 * opt.hidden_size)
        self.cpnet_inject = opt.cpnet_inject


    def forward(self, encoder_out, entity_mask, verb_mask, sentence_mask, cpnet_triples, cpnet_rep):
        """
        Args:
            encoder_out: output of the encoder, size (batch, max_tokens, 2 * hidden_size)
            entity_mask: size (batch, max_sents, max_tokens)
            verb_mask: size (batch, max_sents, max_tokens)
            sentence_mask: size(batch, max_sents, max_tokens)
            cpnet_triples: List, (batch, num_cpnet)
        """
        batch_size = encoder_out.size(0)
        max_sents = entity_mask.size(-2)

        # (batch, max_sents, 4 * hidden_size)
        decoder_in = self.get_masked_input(encoder_out, entity_mask, verb_mask, batch_size = batch_size)
        # (batch, max_sents, 4 * hidden_size)
        attn_probs = None
        if self.cpnet_inject in ['state', 'both']:
            decoder_in, attn_probs = self.CpnetMemory(encoder_out, decoder_in, entity_mask,
                                          sentence_mask, cpnet_triples, cpnet_rep)
        decoder_out, _ = self.Decoder(decoder_in)  # (batch, max_sents, 2 * hidden_size), forward & backward concatenated
        decoder_out = self.Dropout(decoder_out)
        tag_logits = self.Hidden2Tag(decoder_out)  # (batch, max_sents, num_tags)
        assert tag_logits.size() == (batch_size, max_sents, NUM_STATES)

        return tag_logits, attn_probs
    

    def get_masked_input(self, encoder_out, entity_mask, verb_mask, batch_size: int):
        """
        If the entity does not exist in this sentence (entity_mask is all-zero),
        then replace it with an all-zero vector;
        Otherwise, concat the average embeddings of entity and verb
        """
        assert entity_mask.size() == verb_mask.size()
        assert entity_mask.size(-1) == encoder_out.size(-2)

        max_sents = entity_mask.size(-2)
        entity_rep = self.get_masked_mean(source = encoder_out, mask = entity_mask, batch_size = batch_size)  # (batch, max_sents, 2 * hidden_size)
        verb_rep = self.get_masked_mean(source = encoder_out, mask = verb_mask, batch_size = batch_size)  # (batch, max_sents, 2 * hidden_size)
        concat_rep = torch.cat([entity_rep, verb_rep], dim = -1)  # (batch, max_sents, 4 * hidden_size)

        assert concat_rep.size() == (batch_size, max_sents, 4 * self.hidden_size)
        entity_existence = find_allzero_rows(vector = entity_mask).unsqueeze(dim = -1)  # (batch, max_sents, 1)
        masked_rep = concat_rep.masked_fill(mask = entity_existence, value = 0)
        assert masked_rep.size() == (batch_size, max_sents, 4 * self.hidden_size)

        return masked_rep


    def get_masked_mean(self, source, mask, batch_size: int):
        """
        Args:
            source - input tensors, size(batch, tokens, 2 * hidden_size)
            mask - binary masked vectors, size(batch, sents, tokens)
        Return:
            the average of unmasked input tensors, size (batch, sents, 2 * hidden_size)
        """
        max_sents = mask.size(-2)

        bool_mask = (mask.unsqueeze(dim = -1) == 0)  # turn binary masks to boolean values
        masked_source = source.unsqueeze(dim = 1).masked_fill(bool_mask, value = 0)
        masked_source = torch.sum(masked_source, dim = -2)  # sum the unmasked vectors
        assert masked_source.size() == (batch_size, max_sents, 2 * self.hidden_size)

        num_unmasked_tokens = torch.sum(mask, dim = -1, keepdim = True)  # compute the denominator of average op
        masked_mean = torch.div(input = masked_source, other = num_unmasked_tokens)  # average the unmasked vectors

        # division op may cause nan while encoutering 0, so replace nan with 0
        is_nan = torch.isnan(masked_mean)
        masked_mean = masked_mean.masked_fill(is_nan, value = 0)

        assert masked_mean.size() == (batch_size, max_sents, 2 * self.hidden_size)
        return masked_mean


class LocationPredictor(nn.Module):
    """
    Location prediction decoder: sentence-level Bi-LSTM + linear + softmax
    """
    def __init__(self, opt: argparse.Namespace):

        super(LocationPredictor, self).__init__()
        self.hidden_size = opt.hidden_size
        self.Decoder = nn.LSTM(input_size = 4 * opt.hidden_size, hidden_size = opt.hidden_size,
                                    num_layers = 1, batch_first = True, bidirectional = True)
        self.Dropout = nn.Dropout(p = opt.dropout)
        self.Hidden2Score = Linear(d_in = 2 * opt.hidden_size, d_out = 1, dropout = 0)
        self.CpnetMemory = CpnetMemory(opt, query_size=4 * opt.hidden_size, input_size=4 * opt.hidden_size)
        self.cpnet_inject = opt.cpnet_inject


    def forward(self, encoder_out, entity_mask, loc_mask, sentence_mask, cpnet_triples, cpnet_rep):
        """
        Args:
            encoder_out: output of the encoder, size (batch, max_tokens, 2 * hidden_size)
            entity_mask: size (batch, max_sents, max_tokens)
            sentence_mask: size(batch, max_sents, max_tokens)
            cpnet_triples: List, (batch, num_cpnet)
            loc_mask: size (batch, max_cands, max_sents, max_tokens)
        """
        batch_size = encoder_out.size(0)
        max_cands = loc_mask.size(-3)
        max_sents = loc_mask.size(-2)

        decoder_in = self.get_masked_input(encoder_out, entity_mask, loc_mask, batch_size = batch_size)
        decoder_in = decoder_in.view(batch_size * max_cands, max_sents, 4 * self.hidden_size)
        attn_probs = None
        if self.cpnet_inject in ['location', 'both']:
            decoder_in, attn_probs = self.CpnetMemory(encoder_out=NCETModel.expand_dim_3d(encoder_out, max_cands),
                                          decoder_in=decoder_in,
                                          entity_mask=NCETModel.expand_dim_3d(entity_mask, max_cands),
                                          sentence_mask=NCETModel.expand_dim_3d(sentence_mask, max_cands),
                                          cpnet_triples=cpnet_triples,
                                          cpnet_rep=cpnet_rep,
                                          loc_mask = loc_mask.view(batch_size*max_cands, max_sents, -1))
        decoder_out, _ = self.Decoder(decoder_in)  # (batch, max_sents, 2 * hidden_size), forward & backward concatenated
        assert decoder_out.size() == (batch_size * max_cands, max_sents, 2 * self.hidden_size)

        decoder_out = decoder_out.view(batch_size, max_cands, max_sents, 2 * self.hidden_size)
        decoder_out = self.Dropout(decoder_out)
        loc_logits = self.Hidden2Score(decoder_out).squeeze(dim = -1)  # (batch, max_cands, max_sents)

        return loc_logits, attn_probs

    def get_masked_input(self, encoder_out, entity_mask, loc_mask, batch_size: int):
        """
        Concat the mention positions of the entity and each location candidate
        """
        assert entity_mask.size(-1) == loc_mask.size(-1) == encoder_out.size(-2)
        assert entity_mask.size(-2) == loc_mask.size(-2)

        max_cands = loc_mask.size(-3)
        max_sents = loc_mask.size(-2)

        # (batch, max_sents, 2 * hidden_size)
        entity_rep = self.get_masked_mean(source = encoder_out, mask = entity_mask, batch_size = batch_size)
        # (batch, max_cands, max_sents, 2 * hidden_size)
        loc_rep = self.get_masked_loc_mean(source = encoder_out, mask = loc_mask, batch_size = batch_size)
        entity_rep = entity_rep.unsqueeze(dim = 1).expand_as(loc_rep)
        assert entity_rep.size() == loc_rep.size() == (batch_size, max_cands, max_sents, 2 * self.hidden_size)

        concat_rep = torch.cat([entity_rep, loc_rep], dim = -1)
        assert concat_rep.size() == (batch_size, max_cands, max_sents, 4 * self.hidden_size)

        return concat_rep


    def get_masked_loc_mean(self, source, mask, batch_size: int):
        """
        Args:
            source - input tensors, size(batch, tokens, 2 * hidden_size)
            mask - binary masked vectors, size(batch, cands, sents, tokens)
        Return:
            the average of unmasked input tensors, size (batch, cands, sents, 2 * hidden_size)
        """
        max_sents = mask.size(-2)
        max_cands = mask.size(-3)

        bool_mask = (mask.unsqueeze(dim = -1) == 0)  # turn binary masks to boolean values
        source = source.unsqueeze(dim = 1).unsqueeze(dim = 1)  # expand source to (batch, 1, 1, tokens, 2*hidden)
        masked_source = source.masked_fill(bool_mask, value = 0)  # (batch, cands, sents, tokens, 2*hidden)
        masked_source = torch.sum(masked_source, dim = -2)  # (batch, cands, sents, 2*hidden)
        assert masked_source.size() == (batch_size, max_cands, max_sents, 2 * self.hidden_size)

        num_unmasked_tokens = torch.sum(mask, dim = -1, keepdim = True)  # (batch, cands, sents, 1)
        masked_mean = torch.div(input = masked_source, other = num_unmasked_tokens)  # (batch, cands, sents, 2*hidden)

        # division op may cause nan while encoutering 0, so replace nan with 0
        is_nan = torch.isnan(masked_mean)
        masked_mean = masked_mean.masked_fill(is_nan, value = 0)

        assert masked_mean.size() == (batch_size, max_cands, max_sents, 2 * self.hidden_size)
        return masked_mean


    def get_masked_mean(self, source, mask, batch_size: int):
        """
        Args:
            source - input tensors, size(batch, tokens, 2 * hidden_size)
            mask - binary masked vectors, size(batch, sents, tokens)
        Return:
            the average of unmasked input tensors, size (batch, sents, 2 * hidden_size)
        """
        max_sents = mask.size(-2)

        bool_mask = (mask.unsqueeze(dim = -1) == 0)  # turn binary masks to boolean values
        masked_source = source.unsqueeze(dim = 1).masked_fill(bool_mask, value = 0)  # for masked tokens, turn its value to 0
        masked_source = torch.sum(masked_source, dim = -2)  # sum the unmasked token representations
        assert masked_source.size() == (batch_size, max_sents, 2 * self.hidden_size)

        num_unmasked_tokens = torch.sum(mask, dim = -1, keepdim = True)  # compute the denominator of average op (number of unmasked tokens)
        masked_mean = torch.div(input = masked_source, other = num_unmasked_tokens)  # average the unmasked vectors

        # division op may cause nan while encoutering 0, so replace nan with 0
        is_nan = torch.isnan(masked_mean)
        masked_mean = masked_mean.masked_fill(is_nan, value = 0)

        assert masked_mean.size() == (batch_size, max_sents, 2 * self.hidden_size)
        return masked_mean


class CpnetMemory(nn.Module):

    def __init__(self, opt, query_size: int, input_size: int):
        super(CpnetMemory, self).__init__()
        self.cuda = not opt.no_cuda
        self.hidden_size = opt.hidden_size
        self.query_size = query_size
        self.value_size = 2 * opt.hidden_size
        self.input_size = input_size
        self.AttnUpdate = GatedAttnUpdate(query_size=self.query_size, value_size=self.value_size,
                                          input_size=self.input_size, dropout=opt.dropout)


    def forward(self, encoder_out, decoder_in, entity_mask, sentence_mask, cpnet_triples: List[List[str]],
                cpnet_rep, loc_mask = None):
        """
        Args:
            encoder_out: size (batch, max_tokens, 2 * hidden_size)
            decoder_in: (batch, max_sents, 4 * hidden_size) for state tracking,
                        (batch * max_cands, max_sents, 4 * hidden_size) for location prediciton
            entity_mask: size(batch, max_sents, max_tokens)
            sentence_mask: size(batch, max_sents, max_tokens)
            cpnet_triples: List, (batch, num_cands)
        """
        assert encoder_out.size(0) == decoder_in.size(0) == entity_mask.size(0) == \
                sentence_mask.size(0)
        batch_size = encoder_out.size(0)
        ori_batch_size = cpnet_rep.size(0)

        # use the embedding of the current sentence as the attention query
        # (batch, max_sents, 2 * hidden_size)
        # query = self.get_masked_mean(source=encoder_out, mask=sentence_mask, batch_size=batch_size)
        query = decoder_in
        attn_mask = self.get_attn_mask(cpnet_triples)
        if self.cuda:
            attn_mask = attn_mask.cuda()

        if loc_mask is not None:
            assert cpnet_rep.size(0) != batch_size, batch_size % cpnet_rep.size(0) == 0
            max_cands = batch_size // cpnet_rep.size(0)
            cpnet_rep = NCETModel.expand_dim_3d(cpnet_rep, loc_cands=max_cands)
            attn_mask = NCETModel.expand_dim_2d(attn_mask, loc_cands=max_cands)
        update_in, attn_probs = self.AttnUpdate(query=query, values=cpnet_rep, ori_input=decoder_in, attn_mask=attn_mask,
                                    ori_batch_size = ori_batch_size)

        # mask_vec = torch.sum(entity_mask, dim=-1, keepdim=True)
        # if loc_mask is not None:
        #     mask_vec += torch.sum(loc_mask, dim=-1, keepdim=True)
        # update_in = update_in.masked_fill(mask_vec==0, value=0)

        return update_in, attn_probs


    def get_masked_mean(self, source, mask, batch_size: int):
        """
        Args:
            source - input tensors, size(batch, tokens, 2 * hidden_size)
            mask - binary masked vectors, size(batch, sents, tokens)
        Return:
            the average of unmasked input tensors, size (batch, sents, 2 * hidden_size)
        """
        max_sents = mask.size(-2)

        bool_mask = (mask.unsqueeze(dim = -1) == 0)  # turn binary masks to boolean values
        masked_source = source.unsqueeze(dim = 1).masked_fill(bool_mask, value = 0)  # for masked tokens, turn its value to 0
        masked_source = torch.sum(masked_source, dim = -2)  # sum the unmasked token representations
        assert masked_source.size() == (batch_size, max_sents, 2 * self.hidden_size)

        num_unmasked_tokens = torch.sum(mask, dim = -1, keepdim = True)  # compute the denominator of average op (number of unmasked tokens)
        masked_mean = torch.div(input = masked_source, other = num_unmasked_tokens)  # average the unmasked vectors

        # division op may cause nan while encoutering 0, so replace nan with 0
        is_nan = torch.isnan(masked_mean)
        masked_mean = masked_mean.masked_fill(is_nan, value = 0)

        assert masked_mean.size() == (batch_size, max_sents, 2 * self.hidden_size)
        return masked_mean


    def get_attn_mask(self, cpnet_triples: List):
        attn_mask = []
        for instance in cpnet_triples:
            attn_mask.append(list(map(lambda x: x != '', instance)))
        return torch.tensor(attn_mask, dtype=torch.int)



class GatedAttnUpdate(nn.Module):
    """
    Attention + gate update
    """

    def __init__(self, query_size: int, value_size: int, input_size: int, dropout: float):
        super(GatedAttnUpdate, self).__init__()
        self.query_size = query_size
        self.value_size = value_size
        self.input_size = input_size

        attn_vec = torch.empty(query_size, value_size)
        nn.init.xavier_normal_(attn_vec)
        self.attn_vec = nn.Parameter(attn_vec, requires_grad=True)

        self.gate_fc = Linear(input_size + value_size, input_size, dropout=dropout)
        self.concat_fc = Linear(input_size + value_size, input_size, dropout=dropout)
        self.Dropout = nn.Dropout(p=dropout)

        self.attn_log = []

    def forward(self, query, values, ori_input, attn_mask, ori_batch_size: int):
        """
        :param query: (batch, max_sents, query_size)
        :param values: (batch, num_cands, value_size)
        :param attn_mask: (batch, num_cands), 0 for pad values
        :param ori_input: (batch, max_sents, input_size), input vector to be merged with context vector
        :return:
        """
        assert query.size(0) == values.size(0), query.size(1) == ori_input.size(1)
        assert len(query.size()) == len(values.size()) == len(ori_input.size()) == 3
        batch_size = query.size(0)
        num_cands = values.size(1)
        max_sents = query.size(1)
        assert query.size(-1) == self.query_size
        assert values.size(-1) == self.value_size
        assert ori_input.size(-1) == self.input_size

        # attention
        attn_vec = self.attn_vec.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, query_size, value_size)
        # similarity score, (batch, max_sents, num_cands)
        S = torch.bmm(torch.bmm(query, attn_vec), values.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            S = S.masked_fill(attn_mask == 0, float('-inf'))
        probs = F.softmax(S, dim=-1)  # attention weights, (batch, max_sents, num_cands)
        if ori_batch_size != batch_size:
            self.attn_log.extend(probs.view(ori_batch_size, -1, max_sents, num_cands).tolist())
        else:
            self.attn_log.extend(probs.tolist())
        is_nan = torch.isnan(probs)
        probs = probs.masked_fill(is_nan, value=0)  # if no valid triple exist, the system will output nan
        C = torch.bmm(probs, values).squeeze(dim=-1)  # weighted sum, (batch, max_sents, value_size)
        assert C.size() == (batch_size, max_sents, self.value_size)

        # select attention weights for attention loss
        # for location prediction, only select one case since they are the same
        if ori_batch_size != batch_size:
            select_probs = probs.view(ori_batch_size, -1, max_sents, num_cands)
        else:
            select_probs = probs

        # gate
        concat_vec = torch.cat([ori_input, C], dim=-1)
        gate_vec = torch.sigmoid(self.gate_fc(concat_vec))
        cand_input = self.concat_fc(concat_vec)
        final_input = torch.mul(gate_vec, cand_input) + torch.mul(1 - gate_vec, ori_input)
        assert final_input.size() == (batch_size, max_sents, self.input_size)

        return final_input, select_probs


class FixedSentEncoder(nn.Module):
    """
    A encoder that acquires sentence embedding from a fixed pretrained language model
    """
    def __init__(self, opt):
        super(FixedSentEncoder, self).__init__()
        self.embed_size = MODEL_HIDDEN[opt.plm_model_name]
        self.hidden_size = opt.hidden_size
        self.lm_batch_size = opt.batch_size
        self.LSTM = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)

        self.cuda = not opt.no_cuda
        self.Dropout = nn.Dropout(p=opt.dropout)


    def forward(self, input: List[List[str]], tokenizer, encoder):
        """
        Args:
            input: size(batch, num_cands), each is a list of untokenized strings.
        """
        batch_size = len(input)
        num_cands = len(input[0])
        all_sents = itertools.chain.from_iterable(input)  # batch * num_cands
        input_ids = list(map(lambda s: tokenizer.encode(s, add_special_tokens=True), all_sents))
        input_batches = [input_ids[batch_idx * self.lm_batch_size: (batch_idx + 1) * self.lm_batch_size]
                         for batch_idx in range(len(input_ids) // self.lm_batch_size + 1)]
        sent_embed = []

        for batch_input_ids in input_batches:
            mini_batch_size = len(batch_input_ids)
            if not batch_input_ids:
                continue

            batch_input_ids, attention_mask, max_len = \
                FixedSentEncoder.pad_to_longest(batch=batch_input_ids,
                                                pad_id=tokenizer.pad_token_id)
            if self.cuda:
                batch_input_ids = batch_input_ids.cuda()
                attention_mask = attention_mask.cuda()

            with torch.no_grad():
                outputs = encoder(batch_input_ids, attention_mask=attention_mask)

            last_hidden = outputs[0]  # (batch, seq_len, hidden_size)
            assert last_hidden.size() == (mini_batch_size, max_len, self.embed_size)

            encoder_out, _ = self.LSTM(last_hidden)
            encoder_out = self.Dropout(encoder_out)

            for i in range(last_hidden.size(0)):
                embedding = encoder_out[i]  # (max_length, hidden_size)
                pad_mask = attention_mask[i]
                num_tokens = torch.sum(pad_mask) - 2  # number of tokens except <PAD>, <CLS>, <SEP>
                token_embed = embedding[1 : num_tokens + 1]  # get rid of <CLS> (first token) and <SEP> (last token)
                mean_embed = torch.mean(token_embed, dim=0)
                is_nan = torch.isnan(mean_embed)
                mean_embed = mean_embed.masked_fill(is_nan, value=0)
                sent_embed.append(mean_embed)

        sent_embed = torch.stack(sent_embed, dim=0)
        assert sent_embed.size() == (batch_size * num_cands, 2 * self.hidden_size)
        sent_embed = self.Dropout(sent_embed)
        sent_embed = sent_embed.view(batch_size, num_cands, 2 * self.hidden_size)

        return sent_embed


    @staticmethod
    def pad_to_longest(batch: List, pad_id: int) -> (torch.LongTensor, torch.FloatTensor):
        """
        Pad the sentences to the longest length in a batch
        """
        batch_size = len(batch)
        max_length = max([len(batch[i]) for i in range(batch_size)])

        pad_batch = [batch[i] + [pad_id for _ in range(max_length - len(batch[i]))] for i in range(batch_size)]
        pad_batch = torch.tensor(pad_batch, dtype=torch.long)
        # avoid computing attention on padding tokens
        attention_mask = torch.ones_like(pad_batch).masked_fill(mask=(pad_batch==pad_id), value=0)
        assert pad_batch.size() == attention_mask.size() == (batch_size, max_length)

        return pad_batch, attention_mask, max_length


class Linear(nn.Module):
    """
    Simple Linear layer with xavier init
    """
    def __init__(self, d_in: int, d_out: int, dropout: float, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(p = dropout)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return self.dropout(self.linear(x))


class Bilinear(nn.Module):
    """
    Simple Bilinear layer with xavier init and dropout
    """
    def __init__(self, in_1: int, in_2: int, d_out: int, dropout: float, bias: bool = True):
        super(Bilinear, self).__init__()
        self.bilinear = nn.Bilinear(in1_features=in_1, in2_features=in_2, out_features=d_out, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.xavier_normal_(self.bilinear.weight)

    def forward(self, x1, x2):
        return self.dropout(self.bilinear(x1, x2))
