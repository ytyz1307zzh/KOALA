'''
 @Date  : 01/14/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import argparse
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_data', type=str, default='./wiki_para.json', help='data candidates, rough retrieval from wiki')
    parser.add_argument('-output', type=str, default='./result/retrieval_tfidf.json', help='output file to store results')
    parser.add_argument('-num_cands', type=int, default=20,
                        help='number of candidate wiki paragraphs previously retrieved by the TF-IDF ranker.')
    parser.add_argument('-max_num', type=int, default=5, help='top k candidates to keep')
    opt = parser.parse_args()

    raw_data = json.load(open(opt.raw_data, 'r', encoding='utf8'))
    print(f'{len(raw_data)} instances of data loaded.')
    output = []

    for inst_idx in tqdm(range(len(raw_data))):
        instance = raw_data[inst_idx]
        prompt = instance['prompt']
        para_id = instance['para_id']
        topic = instance['topic']
        paragraph = instance['paragraph']
        wiki_cands = instance['wiki']  # a list of wiki paragraphs

        assert len(wiki_cands) == opt.num_cands

        output.append({'id': para_id,
                       'topic': topic,
                       'prompt': prompt,
                       'paragraph': paragraph,
                       'wiki': wiki_cands[:opt.max_num]
                       })

    json.dump(output, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
    print('Finished.')


if __name__ == '__main__':
    main()
