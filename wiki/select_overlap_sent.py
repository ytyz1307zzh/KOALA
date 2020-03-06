'''
 @Date  : 03/02/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import json
import argparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from ConceptNet.Stemmer import PorterStemmer
from transformers import BertTokenizer
stemmer = PorterStemmer()
from spacy.lang.en import STOP_WORDS
from typing import List


def select_noun(sentence: str) -> List[str]:
    token_list = word_tokenize(sentence)
    pos_list = pos_tag(token_list)
    noun_list = []
    for token, pos in pos_list:
        if pos.startswith('NN'):
            noun_list.append(token)
    return noun_list


def check_overlap(query: str, sentence: str) -> bool:
    query_nouns = select_noun(query)
    sent_nouns = select_noun(sentence)
    query_words = set(map(stemmer.stem, query_nouns))
    sent_words = set(map(stemmer.stem, sent_nouns))

    if query_words.intersection(sent_words):
        return True
    else:
        return False



def find_overlap_sentence(query: str, paragraph: str):
    sentences = sent_tokenize(paragraph)
    sentences = [' '.join(word_tokenize(sent)) for sent in sentences]
    overlap_sents = []

    for sent in sentences:
        if check_overlap(query, sent):
            overlap_sents.append(sent)

    return overlap_sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default='./result/retrieval_roberta_large_secondlast_30.json')
    parser.add_argument('-output', type=str, default='./result/retrieval_overlap_sents.json')
    opt = parser.parse_args()

    raw_data = json.load(open(opt.input, 'r', encoding='utf8'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    result = []
    total_tokens = 0
    total_wikis = 0
    overflow_wikis = 0
    max_length = 0

    for instance in tqdm(raw_data):

        para_id = instance['id']
        topic = instance['topic']
        prompt = instance['prompt']
        paragraph = instance['paragraph']
        wiki_list = [x['text'] for x in instance['wiki']]
        query = prompt + ' ' + paragraph

        select_wiki = instance['wiki']
        num_wiki = len(select_wiki)
        for i in range(num_wiki):
            overlap_sents = find_overlap_sentence(query=query, paragraph=wiki_list[i])
            overlap_para = ' '.join(overlap_sents)
            select_wiki[i]['text'] = overlap_para

            input_seq = tokenizer.encode(paragraph, overlap_para, add_special_tokens=True)
            total_tokens += len(input_seq)
            total_wikis += 1
            if len(input_seq) > tokenizer.max_len:
                overflow_wikis += 1
            if len(input_seq) > max_length:
                max_length = len(input_seq)


        result.append({'id': para_id,
                       'topic': topic,
                       'prompt': prompt,
                       'paragraph': paragraph,
                       'wiki': select_wiki
                       })

    json.dump(result, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
    print(f'Average number of tokens: {total_tokens / total_wikis}')
    print(f'Overflow instances: {overflow_wikis}/{total_wikis} ({overflow_wikis / total_wikis * 100:.2f})')
    print(f'Max sequence length: {max_length}')


if __name__ == '__main__':
    main()
