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


def get_sentence_embed(tokenizer, model, sentence: str, cuda: bool):
    sent_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)], dtype=torch.long)  # batch_size = 1
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
    sent_embed, _ = torch.max(sent_embed, dim=0)  # max pooling

    return sent_embed


def select_topk_doc(tokenizer, model, query: str, docs: List[str], max_num: int, cuda: bool) -> (List[int], List[int]):
    """
    Select the k most similar wiki paragraphs to the given query, based on doc embedding.
    """
    doc_embed = []
    for doc in docs:
        doc_embed.append(get_sentence_embed(tokenizer, model, sentence=doc, cuda=cuda))

    query_embed = get_sentence_embed(tokenizer, model, sentence=query, cuda=cuda)
    similarity = [cos_similarity(query_embed, embed) for embed in doc_embed]
    topk_score, topk_id = torch.tensor(similarity).topk(k=max_num, largest=True, sorted=True)

    return topk_score.tolist(), topk_id.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./wiki_para.json', help='path to the english conceptnet')
    parser.add_argument('-output', type=str, default='./result/retrieval_embed.json', help='path to store the generated graph')
    parser.add_argument('-max', type=int, default=5, help='how many triples to collect')
    parser.add_argument('-no_cuda', default=False, action='store_true', help='if specified, then only use cpu')
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
                                  docs=docs, max_num=opt.max, cuda=cuda)

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

