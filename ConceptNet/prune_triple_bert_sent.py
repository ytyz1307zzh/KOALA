'''
 @Date  : 01/09/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import argparse
import json
from tqdm import tqdm
from typing import List, Dict
import re
import time
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
BERT_BASE_HIDDEN = 768


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
    assert vec1.size() == vec2.size() == (BERT_BASE_HIDDEN,)
    return F.cosine_similarity(vec1, vec2, dim=0)


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

def select_triple(tokenizer, model, raw_triples: List[str], paragraph: str,
                  batch_size: int, max: int, cuda: bool) -> (List[str], List):
    """
    Select related triples from the rough retrieval set using BERT embedding.
    Args:
        tokenizer: a BertTokenizer instance.
        model: a BertModel instance.
        raw_triples: triples collected from ConceptNet. Should be a list of strings, which contains 10 fields connected by comma.
        paragraph: the input paragraph.
        max: number of triples to collect.
    Return:
        the top-max similar triples to the context.
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
        last_embed = hidden_states[-1]  # use the embedding from last BERT layer
        assert last_embed.size() == (batch.size(0), batch.size(1), BERT_BASE_HIDDEN)

        for i in range(batch.size(0)):
            embedding = last_embed[i]  # (max_length, hidden_size)
            pad_mask = attention_mask[i]
            num_tokens = torch.sum(pad_mask) - 2  # number of tokens except <PAD>, <CLS>, <SEP>
            token_embed = embedding[1 : num_tokens + 1]  # get rid of <CLS> (first token) and <SEP> (last token)
            triple_embed.append(torch.mean(token_embed, dim=0))

    # now handle the embedding of the paragraph
    para_ids = torch.tensor([tokenizer.encode(paragraph, add_special_tokens=True)])  # batch_size = 1
    if cuda:
        para_ids = para_ids.cuda()
    with torch.no_grad():
        outputs = model(para_ids)

    assert len(outputs) == 3
    _, _, hidden_states = outputs
    assert len(hidden_states) == 13
    last_embed = hidden_states[-1]  # use the embedding from last BERT layer
    assert last_embed.size() == (1, para_ids.size(1), BERT_BASE_HIDDEN)
    para_embed = torch.mean(last_embed[0, 1:-1, :], dim=0)  # get rid of <CLS> (first token) and <SEP> (last token)

    global useful_cnt
    useful_triples_id = [idx for idx in range(total_triples) if get_weight(raw_triples[idx]) >= 1.0]

    # if the number of triples with weight >= 1.0 is large enough, then only select from these triples
    if len(useful_triples_id) >= max:
        useful_triples_embed = [triple_embed[idx] for idx in useful_triples_id]
        similarity = [cos_similarity(para_embed, embed) for embed in useful_triples_embed]
        topk_score, topk_id = torch.tensor(similarity).topk(k=min(max, len(similarity)), largest=True, sorted=True)
        selected_triples = [raw_triples[useful_triples_id[int(idx)]] for idx in topk_id]
        selected_triples = [selected_triples[j] + f', {topk_score[j].item():.4f}' for j in
                            range(len(selected_triples))]  # append score to triple
        useful_cnt += 1

    else:
        similarity = [cos_similarity(para_embed, embed) for embed in triple_embed]
        topk_score, topk_id = torch.tensor(similarity).topk(k = min(max, len(similarity)), largest = True, sorted = True)
        selected_triples = [raw_triples[int(idx)] for idx in topk_id]
        selected_triples = [selected_triples[j]+f', {topk_score[j].item():.4f}' for j in
                            range(len(selected_triples))]  # append score to triple

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

        # raw_triples may contain repetitive fields (multiple entities)
        selected_triples = select_triple(tokenizer = tokenizer, model = model, raw_triples = list(set(raw_triples)),
                                         paragraph = paragraph, batch_size = opt.batch, max = opt.max, cuda = cuda)

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