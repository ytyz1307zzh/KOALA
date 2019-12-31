'''
 @Date  : 12/11/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import torch
import torch.nn as nn
import json
import os
import time
import numpy as np
from typing import List, Dict
from Constants import *
from utils import *
from allennlp.modules.elmo import Elmo
from torchcrf import CRF
import argparse


class NCETModel(nn.Module):

    def __init__(self, opt: argparse.Namespace, is_test: bool):

        super(NCETModel, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.embed_size = opt.embed_size

        self.EmbeddingLayer = NCETEmbedding(embed_size = opt.embed_size, elmo_dir = opt.elmo_dir,
                                            dropout = opt.dropout, elmo_dropout = opt.elmo_dropout)
        self.TokenEncoder = nn.LSTM(input_size = opt.embed_size, hidden_size = opt.hidden_size,
                                    num_layers = 1, batch_first = True, bidirectional = True)
        self.Dropout = nn.Dropout(p = opt.dropout)

        # state tracking modules
        self.StateTracker = StateTracker(hidden_size = opt.hidden_size, dropout = opt.dropout)
        self.CRFLayer = CRF(NUM_STATES, batch_first = True)

        # location prediction modules
        self.LocationPredictor = LocationPredictor(hidden_size = opt.hidden_size, dropout = opt.dropout)
        self.CrossEntropy = nn.CrossEntropyLoss(ignore_index = PAD_LOC, reduction = 'mean')

        self.is_test = is_test
        

    def forward(self, char_paragraph: torch.Tensor, entity_mask: torch.IntTensor, verb_mask: torch.IntTensor,
                loc_mask: torch.IntTensor, gold_loc_seq: torch.IntTensor, gold_state_seq: torch.IntTensor,
                num_cands: torch.IntTensor):
        """
        Args:
            gold_loc_seq: size (batch, max_sents)
            gold_state_seq: size (batch, max_sents)
            num_cands: size(batch,)
        """
        assert entity_mask.size(-2) == verb_mask.size(-2) == loc_mask.size(-2) == gold_state_seq.size(-1) == gold_loc_seq.size(-1)
        assert entity_mask.size(-1) == verb_mask.size(-1) == loc_mask.size(-1) == char_paragraph.size(-2)
        batch_size = char_paragraph.size(0)
        max_tokens = char_paragraph.size(1)
        max_sents = gold_state_seq.size(-1)
        max_cands = loc_mask.size(-3)

        embeddings = self.EmbeddingLayer(char_paragraph, verb_mask)  # (batch, max_tokens, embed_size)
        token_rep, _ = self.TokenEncoder(embeddings)  # (batch, max_tokens, 2*hidden_size)
        token_rep = self.Dropout(token_rep)
        assert token_rep.size() == (batch_size, max_tokens, 2 * self.hidden_size)

        # state cheng prediction
        # size (batch, max_sents, NUM_STATES)
        tag_logits = self.StateTracker(encoder_out = token_rep, entity_mask = entity_mask, verb_mask = verb_mask)
        tag_mask = (gold_state_seq != PAD_STATE) # mask the padded part so they won't count in loss
        log_likelihood = self.CRFLayer(emissions = tag_logits, tags = gold_state_seq.long(), mask = tag_mask, reduction = 'token_mean')

        state_loss = -log_likelihood  # State classification loss is negative log likelihood
        pred_state_seq = self.CRFLayer.decode(emissions=tag_logits, mask=tag_mask)
        assert len(pred_state_seq) == batch_size
        correct_state_pred, total_state_pred = compute_state_accuracy(pred=pred_state_seq, gold=gold_state_seq.tolist(),
                                                        pad_value=PAD_STATE)

        # location prediction
        # size (batch, max_cands, max_sents)
        loc_logits = self.LocationPredictor(encoder_out = token_rep, entity_mask = entity_mask, loc_mask = loc_mask)
        loc_logits = loc_logits.transpose(-1, -2)  # size (batch, max_sents, max_cands)
        masked_loc_logits = self.mask_loc_logits(loc_logits = loc_logits, num_cands = num_cands)  # (batch, max_sents, max_cands)
        masked_gold_loc_seq = self.mask_undefined_loc(gold_loc_seq = gold_loc_seq, mask_value = PAD_LOC)  # (batch, max_sents)
        loc_loss = self.CrossEntropy(input = masked_loc_logits.view(batch_size * max_sents, max_cands),
                                     target = masked_gold_loc_seq.view(batch_size * max_sents).long())
        correct_loc_pred, total_loc_pred = compute_loc_accuracy(logits = masked_loc_logits, gold = masked_gold_loc_seq,
                                                                pad_value = PAD_LOC)
        # assert total_loc_pred > 0

        if self.is_test:  # inference
            pred_loc_seq = get_pred_loc(loc_logits = masked_loc_logits, gold_loc_seq = gold_loc_seq)
            return pred_state_seq, pred_loc_seq, correct_state_pred, total_state_pred, correct_loc_pred, total_loc_pred

        return state_loss, loc_loss, correct_state_pred, total_state_pred, correct_loc_pred, total_loc_pred


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

    
class NCETEmbedding(nn.Module):

    def __init__(self, embed_size: int, elmo_dir: str, dropout: float, elmo_dropout: float):

        super(NCETEmbedding, self).__init__()
        self.embed_size = embed_size
        self.options_file = os.path.join(elmo_dir, 'elmo_2x4096_512_2048cnn_2xhighway_options.json')
        self.weight_file = os.path.join(elmo_dir, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
        self.elmo = Elmo(self.options_file, self.weight_file, num_output_representations=1, requires_grad=False,
                            do_layer_norm=False, dropout=elmo_dropout)
        self.embed_project = Linear(1024, self.embed_size - 1, dropout = dropout)  # 1024 is the default size of Elmo, leave 1 dim for verb indicator


    def forward(self, char_paragraph: torch.Tensor, verb_mask: torch.IntTensor):
        """
        Args: 
            char_paragraph - character ids of the paragraph, generated by function "batch_to_ids"
            verb_mask - size (batch, max_sents, max_tokens)
        Return:
            embeddings - token embeddings, size (batch, max_tokens, embed_size)
        """
        batch_size = char_paragraph.size(0)
        max_tokens = char_paragraph.size(1)

        elmo_embeddings = self.get_elmo(char_paragraph, batch_size = batch_size, max_tokens = max_tokens)
        if self.embed_size != 1025:
            elmo_embeddings = self.embed_project(elmo_embeddings)
        verb_indicator = self.get_verb_indicator(verb_mask, batch_size = batch_size, max_tokens = max_tokens)
        embeddings = torch.cat([elmo_embeddings, verb_indicator], dim = -1)

        assert embeddings.size() == (batch_size, max_tokens, self.embed_size)
        return embeddings


    def get_elmo(self, char_paragraph: torch.Tensor, batch_size: int, max_tokens: int):
        """
        Compute the Elmo embedding of the paragraphs.
        Return:
            Elmo embeddings, size(batch, max_tokens, elmo_embed_size=1024)
        """
        # embeddings['elmo_representations'] is a list of tensors with length 'num_output_representations' (here it = 1)
        elmo_embeddings = self.elmo(char_paragraph)['elmo_representations'][0]  # (batch, max_tokens, elmo_embed_size=1024)
        assert elmo_embeddings.size() == (batch_size, max_tokens, 1024)
        return elmo_embeddings

    
    def get_verb_indicator(self, verb_mask: torch.IntTensor, batch_size: int, max_tokens: int):
        """
        Get the binary scalar indicator for each token
        """
        verb_indicator = torch.sum(verb_mask, dim = 1, dtype = torch.float).unsqueeze(dim = -1)
        assert verb_indicator.size() == (batch_size, max_tokens, 1)
        return verb_indicator


class StateTracker(nn.Module):
    """
    State tracking decoder: sentence-level Bi-LSTM + linear + CRF
    """
    def __init__(self, hidden_size: int, dropout: float):

        super(StateTracker, self).__init__()
        self.hidden_size = hidden_size
        self.Decoder = nn.LSTM(input_size = 4 * hidden_size, hidden_size = hidden_size,
                                    num_layers = 1, batch_first = True, bidirectional = True)
        self.Dropout = nn.Dropout(p = dropout)
        self.Hidden2Tag = Linear(d_in = 2 * hidden_size, d_out = NUM_STATES, dropout = 0)


    def forward(self, encoder_out, entity_mask, verb_mask):
        """
        Args:
            encoder_out: output of the encoder, size (batch, max_tokens, 2 * hidden_size)
            entity_mask: size (batch, max_sents, max_tokens)
            verb_mask: size (batch, max_sents, max_tokens)
        """
        batch_size = encoder_out.size(0)
        max_sents = entity_mask.size(-2)

        decoder_in = self.get_masked_input(encoder_out, entity_mask, verb_mask, batch_size = batch_size)  # (batch, max_sents, 4 * hidden_size)
        decoder_out, _ = self.Decoder(decoder_in)  # (batch, max_sents, 2 * hidden_size), forward & backward concatenated
        decoder_out = self.Dropout(decoder_out)
        tag_logits = self.Hidden2Tag(decoder_out)  # (batch, max_sents, num_tags)
        assert tag_logits.size() == (batch_size, max_sents, NUM_STATES)

        return tag_logits
    

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
    def __init__(self, hidden_size: int, dropout: float):

        super(LocationPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.Decoder = nn.LSTM(input_size = 4 * hidden_size, hidden_size = hidden_size,
                                    num_layers = 1, batch_first = True, bidirectional = True)
        self.Dropout = nn.Dropout(p = dropout)
        self.Hidden2Score = Linear(d_in = 2 * hidden_size, d_out = 1, dropout = 0)


    def forward(self, encoder_out, entity_mask, loc_mask):
        """
        Args:
            encoder_out: output of the encoder, size (batch, max_tokens, 2 * hidden_size)
            entity_mask: size (batch, max_sents, max_tokens)
            loc_mask: size (batch, max_cands, max_sents, max_tokens)
        """
        batch_size = encoder_out.size(0)
        max_cands = loc_mask.size(-3)
        max_sents = loc_mask.size(-2)

        decoder_in = self.get_masked_input(encoder_out, entity_mask, loc_mask, batch_size = batch_size)
        decoder_in = decoder_in.view(batch_size * max_cands, max_sents, 4 * self.hidden_size)
        decoder_out, _ = self.Decoder(decoder_in)  # (batch, max_sents, 2 * hidden_size), forward & backward concatenated
        assert decoder_out.size() == (batch_size * max_cands, max_sents, 2 * self.hidden_size)

        decoder_out = decoder_out.view(batch_size, max_cands, max_sents, 2 * self.hidden_size)
        decoder_out = self.Dropout(decoder_out)
        loc_logits = self.Hidden2Score(decoder_out).squeeze(dim = -1)  # (batch, max_cands, max_sents)

        return loc_logits

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
