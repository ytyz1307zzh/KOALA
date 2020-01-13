'''
 @Date  : 01/12/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import argparse
import json
from tqdm import tqdm
from typing import List, Dict
import time
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
BERT_BASE_HIDDEN = 768


def cos_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    assert vec1.size() == vec2.size() == (BERT_BASE_HIDDEN,)
    return F.cosine_similarity(vec1, vec2, dim=0)


def get_span_embed(model, sent_ids: torch.Tensor, cuda: bool):
    if cuda:
        sent_ids = sent_ids.cuda()
    with torch.no_grad():
        outputs = model(sent_ids)

    assert len(outputs) == 3
    _, _, hidden_states = outputs
    assert len(hidden_states) == 13
    last_embed = hidden_states[-1]
    assert last_embed.size() == (1, sent_ids.size(1), BERT_BASE_HIDDEN)
    sent_embed = last_embed[0, 1:-1, :]  # get rid of <CLS> (first token) and <SEP> (last token)

    return sent_embed


def get_max_context(doc_spans, position):
    """
    Find the 'max context' doc span for the token.
    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return best_span_index


def get_sentence_embed(tokenizer, model, sentence: str, cuda: bool,
                       max_len: int, doc_stride: int, pooling: str):
    
    sent_length = len(tokenizer.tokenize(sentence))
    all_spans = []
    all_text_ids = []
    tokens_read = 0

    # split the long document into chunks, if its length > 512
    while True:
        encode_dict = tokenizer.encode_plus(sentence,
                                            add_special_tokens=True,
                                            max_length=max_len,
                                            stride=doc_stride,
                                            pad_to_max_length=False,
                                            return_overflowing_tokens=True)
        token_ids = encode_dict['input_ids']
        num_words = len(token_ids) - tokenizer.num_added_tokens()  # won't count special tokens
        all_text_ids.append(token_ids)
        if tokens_read != 0:
            tokens_read -= doc_stride  # start position of the current span, including overlap
        all_spans.append({'start': tokens_read,
                          'length': num_words
                          })
        tokens_read += num_words

        if 'overflowing_tokens' not in encode_dict:
            break

        sentence = encode_dict['overflowing_tokens']

    # find the best context span for each word
    best_span_idx = []
    for i in range(sent_length):
        best_span_idx.append(get_max_context(all_spans, i))

    # get embeddings of each span from BERT
    span_embed = []
    for input_ids in all_text_ids:
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # batch size = 1
        output_embed = get_span_embed(model, sent_ids=input_ids, cuda=cuda)
        span_embed.append(output_embed)  # word embeddings without

    # according to its best span, collect word embedding
    word_embed = []
    for i in range(sent_length):
        best_span = best_span_idx[i]
        best_embed = span_embed[best_span][i - all_spans[best_span]['start']]
        word_embed.append(best_embed)

    # pooling method to acquire the sentence embedding
    word_embed = torch.stack(word_embed, dim=0)
    assert word_embed.size() == (sent_length, BERT_BASE_HIDDEN)
    if pooling == 'max':
        sent_embed, _ = torch.max(word_embed, dim=0)
    elif pooling == 'mean':
        sent_embed = torch.mean(word_embed, dim=0)
    else:
        raise ValueError('Invalid pooling method!')
    assert sent_embed.size() == (BERT_BASE_HIDDEN,)
    
    return sent_embed


def select_topk_doc(tokenizer, model, query: str, docs: List[str],
                    max_num: int, cuda: bool, doc_stride: int) -> (List[int], List[int]):
    """
    Select the k most similar wiki paragraphs to the given query, based on doc embedding.
    """
    doc_embed = []
    chunk_len = tokenizer.max_len
    for doc in docs:
        doc_embed.append(get_sentence_embed(tokenizer, model, sentence=doc, cuda=cuda, max_len=chunk_len, doc_stride=doc_stride))

    query_embed = get_sentence_embed(tokenizer, model, sentence=query, cuda=cuda, max_len=chunk_len, doc_stride=doc_stride)
    similarity = [cos_similarity(query_embed, embed) for embed in doc_embed]
    topk_score, topk_id = torch.tensor(similarity).topk(k=max_num, largest=True, sorted=True)

    return topk_score.tolist(), topk_id.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./wiki_para.json', help='path to the english conceptnet')
    parser.add_argument('-output', type=str, default='./result/retrieval_embed.json', help='path to store the generated graph')
    parser.add_argument('-max', type=int, default=5, help='how many triples to collect')
    parser.add_argument('-doc_stride', type=int, default=128,
                        help='when splitting up a long document into chunks, how much stride to take between chunks')
    parser.add_argument('-no_cuda', default=False, action='store_true', help='if specified, then only use cpu')
    parser.add_argument('-pooling', default='mean', choices=['max', 'mean'],
                        help='pooling method to aggregate BERT word embeddings into sentence embedding')
    opt = parser.parse_args()

    raw_data = json.load(open(opt.input, 'r', encoding='utf8'))
    cuda = False if opt.no_cuda else True
    result = []

    print('[INFO] Loading pretrained BERT...')
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    if cuda:
        model.cuda()
    print(f'[INFO] Model loaded. Time elapse: {time.time() - start_time:.2f}s')

    for instance in tqdm(raw_data):
        para_id = instance['para_id']
        paragraph = instance['paragraph']
        topic = instance['topic']
        prompt = instance['prompt']
        wiki_cands = instance['wiki']

        docs = [wiki['text'] for wiki in wiki_cands]
        topk_score, topk_id = select_topk_doc(tokenizer=tokenizer, model=model, query=paragraph,
                                              docs=docs, max_num=opt.max, cuda=cuda,
                                              doc_stride=opt.doc_stride, pooling=opt.pooling)

        selected_wiki = [wiki_cands[idx] for idx in topk_id]
        assert len(selected_wiki) == opt.max
        for i in range(opt.max):
            selected_wiki[i]['similarity'] = topk_score[i]

        result.append({'id': para_id,
                       'topic': topic,
                       'prompt': prompt,
                       'paragraph': paragraph,
                       'wiki': selected_wiki
                       })

    json.dump(result, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

