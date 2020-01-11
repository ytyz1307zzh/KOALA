'''
 @Date  : 01/10/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import argparse
import json
from tqdm import tqdm
from typing import List, Dict, Set
import re
import time
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import torch.nn.functional as F
from spacy.lang.en import STOP_WORDS
STOP_WORDS = set(STOP_WORDS) - {'bottom', 'serious', 'top', 'alone', 'around', 'used', 'behind', 'side', 'mine', 'well'}
from Stemmer import PorterStemmer
stemmer = PorterStemmer()
BERT_HIDDEN_SIZE = 768


def stem(word: str) -> str:
    """
    Stem a single word
    """
    word = word.lower().strip()
    return stemmer.stem(word)


def find_entity_in_para(paragraph: str, entity: str) -> Set[str]:
    """
    Find all existing forms of the given entity (might be separated by semicolon) in the certain paragraph.
    """
    para_tokens = [token for token in paragraph.strip().split()]
    entity_tokens = {stem(token) for ent in entity.strip().split(';') for token in ent.strip().split()}
    entity_set = set()

    for token in para_tokens:
        if stem(token) in entity_tokens:
            entity_set.add(token)

    return entity_set


def remove_stopword_and_entity(text: List[str], entity_set: Set[str]) -> List[int]:
    """
    Args:
        text: tokenized paragraph
    Returns:
        ids of non-stop words
    """
    return [idx for idx in range(len(text))
            if text[idx] not in STOP_WORDS
            and text[idx] not in entity_set
            and text[idx].isalpha()]


def get_weight(line: str) -> float:
    triple = line.strip().split(', ')
    assert len(triple) >= 9
    weight = float(triple[7])
    return weight


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


def find_token_offset(origin_tokens: List[str], tokens: List[str]) -> List[int]:
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


def triple2sent(raw_triples: List[str], rel_rules: Dict[str, str], trans_rules: Dict[str, str]):
    """
    Turn the triples to natural language sentences.
    """
    result = []

    for line in raw_triples:

        triple = line.strip().split(', ')
        assert len(triple) == 9

        relation = triple[0]
        subj = triple[2]
        obj = triple[5]
        direction = triple[-1]  # LEFT or RIGHT

        # if the semantic role of the entity (subj or obj) does not match, skip this
        if not valid_direction(relation = relation, direction = direction.lower(), rel_rules = rel_rules):
            continue

        # turn the delimiter from _ to space
        subj = ' '.join(subj.split('_'))
        obj = ' '.join(obj.split('_'))

        sentence = trans_rules[relation]
        sentence = re.sub('A', subj, sentence)
        sentence = re.sub('B', obj, sentence)

        triple.append(sentence)
        result.append(', '.join(triple))

    return result


def cos_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    assert vec1.size() == vec2.size() == (BERT_HIDDEN_SIZE,)
    return F.cosine_similarity(vec1, vec2, dim=0)


def find_neighbor_in_sent(raw_triple: str)-> List[int]:
    """
    Find the positions of neighboring concept in the sentence.
    Returns:
        the list of original token ids of the neighboring concept.
    """
    triple = raw_triple.strip().split(', ')
    assert len(triple) == 10

    subj, obj, direction = triple[2], triple[5], triple[-2]
    if direction == 'LEFT':
        neighbor = obj.strip().split('_')
    elif direction == 'RIGHT':
        neighbor = subj.strip().split('_')

    sentence = triple[-1].strip().split()
    sentence_len = len(sentence)
    neighbor_len = len(neighbor)
    for i in range(sentence_len - neighbor_len + 1):
        if sentence[i : i + neighbor_len] == neighbor:
            return list(range(i, i + neighbor_len))

    raise ValueError('Cannot find the neighbor concept in the sentence')


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

    return pad_batch, attention_mask


useful_cnt = 0

def select_triple(tokenizer, model, raw_triples: List[str], paragraph: str, entity_set: Set[str],
                  batch_size: int, max_num: int, cuda: bool) -> (List[str], List):
    """
    Select related triples from the rough retrieval set using BERT embedding.
    Args:
        tokenizer: a BertTokenizer instance.
        model: a BertModel instance.
        raw_triples: triples collected from ConceptNet. Should be a list of strings, which contains 10 fields connected by comma.
        paragraph: the input paragraph.
        max_num: number of triples to collect.
    Return:
        the top-max_num similar triples to the context.
    """
    total_triples = len(raw_triples)
    cand_sents = list(map(lambda x: x.strip().split(', ')[-1], raw_triples))

    input_ids = list(map(lambda s: tokenizer.encode(s, add_special_tokens=True), cand_sents))
    input_batches = [input_ids[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                     for batch_idx in range(len(input_ids) // batch_size + 1)]

    triple_embed = []  # ConceptNet triple embeddings acquired from BERT

    for batch in input_batches:
        # batch: (batch, seq_len)
        if not batch:
            continue
        batch, attention_mask = pad_to_longest(batch = batch, pad_id = tokenizer.pad_token_id)
        if cuda:
            batch = batch.cuda()
            attention_mask = attention_mask.cuda()
        with torch.no_grad():
            outputs = model(batch, attention_mask=attention_mask)
        assert len(outputs) == 3

        _, _, hidden_states = outputs
        assert len(hidden_states) == 13
        last_embed = hidden_states[-1]  # use the embedding from second-last BERT layer
        assert last_embed.size() == (batch.size(0), batch.size(1), BERT_HIDDEN_SIZE)

        for i in range(batch.size(0)):
            embedding = last_embed[i]  # (max_length, hidden_size)
            pad_mask = attention_mask[i]
            num_tokens = torch.sum(pad_mask) - 2  # number of tokens except <PAD>, <CLS>, <SEP>
            token_embed = embedding[1 : num_tokens + 1]  # get rid of <CLS> (first token) and <SEP> (last token)
            triple_embed.append(token_embed)

    # now handle the embedding of the paragraph
    para_ids = torch.tensor([tokenizer.encode(paragraph, add_special_tokens=True)])  # batch_size = 1
    if cuda:
        para_ids = para_ids.cuda()
    with torch.no_grad():
        outputs = model(para_ids)
    assert len(outputs) == 3
    _, _, hidden_states = outputs
    assert len(hidden_states) == 13
    last_embed = hidden_states[-1]  # use the embedding from second-last BERT layer
    assert last_embed.size() == (1, para_ids.size(1), BERT_HIDDEN_SIZE)
    para_embed = last_embed[0, 1:-1, :]  # get rid of <CLS> (first token) and <SEP> (last token)

    # get embedding list of the content words in paragraph
    para_tokens = paragraph.strip().split()
    raw_content_id = remove_stopword_and_entity(text=para_tokens, entity_set=entity_set)
    raw_content_word = [para_tokens[idx] for idx in raw_content_id]
    offset_map = find_token_offset(origin_tokens=para_tokens,
                                   tokens=tokenizer.convert_ids_to_tokens(para_ids[0], skip_special_tokens=True))
    offset_map = torch.tensor(offset_map, dtype=torch.long)
    if cuda:
        offset_map = offset_map.cuda()
    content_embed = []
    for i in raw_content_id:
        token_mask = (offset_map == i).unsqueeze(-1)  # find the ids of this content word in the tokenized sequence
        token_embed = para_embed.masked_select(mask=token_mask).view(token_mask.sum(), BERT_HIDDEN_SIZE)
        token_embed = torch.mean(token_embed, dim=0)
        content_embed.append(token_embed)

    # get embedding of the neighbor concept in each candidate triple
    similarity = []
    matched_words = []  # the matched word in the context to the neighbor concept
    for i in range(total_triples):
        triple = raw_triples[i]
        offset_map = find_token_offset(origin_tokens=cand_sents[i].strip().split(),
                                       tokens=tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True))
        raw_neighbor_id = find_neighbor_in_sent(raw_triple=triple)  # find the ids of neighbor concept in original sentence
        neighbor_id = torch.tensor([idx for idx in range(len(offset_map)) if offset_map[idx] in raw_neighbor_id],
                                   dtype=torch.long)  # map ids to the tokenized sequence
        if cuda:
            neighbor_id = neighbor_id.cuda()
        embed = triple_embed[i].index_select(dim=0, index=neighbor_id)
        assert embed.size() == (len(neighbor_id), BERT_HIDDEN_SIZE)
        neighbor_embed = torch.mean(embed, dim=0)
        content_sim = [cos_similarity(emb, neighbor_embed).item() for emb in content_embed]
        matched_id = np.argmax(content_sim)
        similarity.append(content_sim[matched_id])
        matched_words.append(raw_content_word[matched_id])

    global useful_cnt
    useful_triples_id = [idx for idx in range(total_triples) if get_weight(raw_triples[idx]) >= 1.0]

    if len(useful_triples_id) >= max_num:
        useful_similarity = [similarity[idx] for idx in useful_triples_id]
        topk_score, topk_id = torch.tensor(useful_similarity).topk(k=max_num, largest=True, sorted=True)
        selected_triples = [raw_triples[useful_triples_id[int(idx)]] for idx in topk_id]
        useful_matched_words = [matched_words[useful_triples_id[int(idx)]] for idx in topk_id]
        selected_triples = [selected_triples[j] + f', {topk_score[j].item():.4f}, {useful_matched_words[j]}'
                            for j in range(len(selected_triples))]  # append score to triple
        useful_cnt += 1

    else:
        topk_score, topk_id = torch.tensor(similarity).topk(k=min(max_num, len(similarity)), largest=True, sorted=True)
        selected_triples = [raw_triples[int(idx)] for idx in topk_id]
        matched_words = [matched_words[int(idx)] for idx in topk_id]
        selected_triples = [selected_triples[j] + f', {topk_score[j].item():.4f}, {matched_words[j]}'
                            for j in range(len(selected_triples))]  # append score to triple

    return selected_triples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./rough_retrieval.json', help='path to the english conceptnet')
    parser.add_argument('-output', type=str, default='./retrieval.json', help='path to store the generated graph')
    parser.add_argument('-relation', type=str, default='./relation_direction.txt', help='path to the relation rules')
    parser.add_argument('-transform', type=str, default='./triple2sent.txt',
                        help='path to the file that describes the rules to transform triples into natural language')
    parser.add_argument('-max', type=int, default=10, help='how many triples to collect')
    parser.add_argument('-batch', type=int, default=128, help='batch size of BERT features')
    parser.add_argument('-no_cuda', default=False, action='store_true', help='if specified, then only use cpu')
    opt = parser.parse_args()

    data = json.load(open(opt.input, 'r', encoding='utf8'))
    rel_rules = read_relation(opt.relation)
    trans_rules = read_transform(opt.transform)
    result = []
    less_cnt = 0
    cuda = False if opt.no_cuda else True

    print('[INFO] Loading pretrained BERT...')
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    if cuda:
        model.cuda()
    print(f'[INFO] Model loaded. Time elapse: {time.time()-start_time:.2f}s')

    for instance in tqdm(data):
        para_id = instance['id']
        entity = instance['entity']
        paragraph = instance['paragraph']
        topic = instance['topic']
        prompt = instance['prompt']
        raw_triples = instance['cpnet']

        # omit the triples with invalid direction and transform the others to natural sentences
        raw_triples = triple2sent(raw_triples = raw_triples, rel_rules = rel_rules, trans_rules = trans_rules)

        entity_set = find_entity_in_para(paragraph=paragraph, entity=entity)

        # raw_triples may contain repetitive fields (multiple entities)
        selected_triples = select_triple(tokenizer = tokenizer, model = model, raw_triples = list(set(raw_triples)),
                                         paragraph = paragraph, entity_set = entity_set,
                                         batch_size = opt.batch, max_num = opt.max, cuda = cuda)

        if len(selected_triples) < opt.max:
            less_cnt += 1

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
    print(f'Instances with less than {opt.max} ConceptNet triples collected: {less_cnt} ({(less_cnt / total_instances) * 100:.2f}%)')
    print(f'Instances with more than {opt.max} ConceptNet triples with weight >=1.0: {useful_cnt} ({(useful_cnt / total_instances) * 100:.2f}%)')