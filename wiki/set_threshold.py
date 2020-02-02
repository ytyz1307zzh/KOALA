'''
 @Date  : 02/02/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import argparse
import json
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, required=True, help='data candidates, rough retrieval from wiki')
    parser.add_argument('-output', type=str, default='./result/retrieval.json', help='output file to store results')
    parser.add_argument('-threshold', type=float, default=0.98,
                        help='candidates with similarity score lower than threshold will be discarded.')
    parser.add_argument('-max_num', type=int, default=5, help='top k candidates to keep')
    opt = parser.parse_args()

    raw_data = json.load(open(opt.input, 'r', encoding='utf8'))
    print(f'{len(raw_data)} instances of data loaded.')
    output = []

    for inst_idx in tqdm(range(len(raw_data))):
        instance = raw_data[inst_idx]
        prompt = instance['prompt']
        para_id = instance['id']
        topic = instance['topic']
        paragraph = instance['paragraph']
        wiki_cands = instance['wiki']  # a list of wiki paragraphs

        assert len(wiki_cands) == opt.max_num

        if 'similarity' in wiki_cands[0].keys():
            new_wiki = [wiki for wiki in wiki_cands if wiki['similarity'] > opt.threshold]
        elif 'cls_score' in wiki_cands[0].keys():
            new_wiki = [wiki for wiki in wiki_cands if wiki['cls_score'] > opt.threshold]

        output.append({'id': para_id,
                       'topic': topic,
                       'prompt': prompt,
                       'paragraph': paragraph,
                       'wiki': new_wiki
                       })

    json.dump(output, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
    print('Finished.')


if __name__ == '__main__':
    main()
