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
    parser.add_argument('-input', type=str, default='./wiki_para.json', help='data candidates, rough retrieval from wiki')
    parser.add_argument('-output', type=str, default='./nq-classifier/dev.json.sent', help='output file to store the processed data')
    opt = parser.parse_args()

    raw_data = json.load(open(opt.input, 'r', encoding='utf8'))
    print(f'{len(raw_data)} instances of data loaded.')
    all_instances = []

    for instance in tqdm(raw_data):
        prompt = instance['prompt']
        para_id = instance['para_id']
        paragraph = instance['paragraph']
        wiki_cands = instance['wiki']  # a list of wiki paragraphs

        all_paragraphs = []
        for wiki_idx in range(len(wiki_cands)):
            wiki_data = wiki_cands[wiki_idx]
            wiki_para_id = wiki_data['para_id']
            wiki_title = wiki_data['wiki_id']
            context = wiki_data['text']

            qas = [{'question': prompt,  # TODO: use the paragraph or the prompt as question?
                    'id': f'{para_id}-{wiki_para_id}',
                    }]

            para_data = {
                'qas': qas,
                'context_para': [[0, len(context)]],
                'para_labels': ['0'],
                'keep_or_not': [True],
                'title': wiki_title,
                'context': context,
            }
            all_paragraphs.append(para_data)

        all_instances.append({'paragraphs': all_paragraphs})

    result = {'data': all_instances}
    json.dump(result, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
    print('Finished.')



if __name__ == "__main__":
    main()