'''
 @Date  : 01/11/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
 Prepare input instances for retrieve_para.py
'''
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-input', type=str, default='../ConceptNet/rough_retrieval.json',
                    help='data file that contain data instances for all datasets, e.g.,'
                         'ConceptNet retrieval file')
parser.add_argument('-output', type=str, default='./wiki_query.json')
opt = parser.parse_args()

data = json.load(open(opt.input, 'r', encoding='utf8'))
result = []
ids = set()

for instance in tqdm(data):
    para_id = instance['id']
    entity = instance['entity']
    paragraph = instance['paragraph']
    topic = instance['topic']
    prompt = instance['prompt']

    if para_id in ids:
        continue

    ids.add(para_id)
    result.append({'id': para_id,
                   'entity': entity,
                   'topic': topic,
                   'prompt': prompt,
                   'paragraph': paragraph,
                   })

json.dump(result, open(opt.output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
print('Number of saved data: ', len(result))
