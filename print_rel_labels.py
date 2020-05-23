'''
 @Date  : 02/23/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
 Print the relevance labels of knowledge triples in each instance
 used in attention loss and store in a TSV file.
'''

import argparse
import json
from tqdm import tqdm
from Dataset import *
from Constants import *
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
torch.set_printoptions(precision=3, edgeitems=6, sci_mode=False)

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='data/test.json')
parser.add_argument('-output', type=str, default='predict/rellabel_test.tsv')
parser.add_argument('-cpnet_path', type=str, default="ConceptNet/result/retrieval.json", help="path to conceptnet triples")
parser.add_argument('-state_verb', type=str, default='ConceptNet/result/state_verb_cut.json', help='path to state verb dict')
parser.add_argument('-batch_size', type=int, default=64)
opt = parser.parse_args()
opt.cpnet_struc_input = False  # add this arg to circumvent errors


def get_output(metadata: Dict, state_rel_labels: List[List[int]], loc_rel_labels: List[List[int]],
               gold_state_seq: List[int], cpnet_triples: List[str]) -> Dict:
    """
    Get the predicted output from generated sequences by the model.
    """
    para_id = metadata['para_id']
    entity_name = metadata['entity']
    total_sents = metadata['total_sents']

    gold_state_seq = [idx2state[idx] for idx in gold_state_seq if idx != PAD_STATE]
    gold_loc_seq = metadata['raw_gold_loc']  # gold locations in string form

    result = {'id': para_id,
              'entity': entity_name,
              'total_sents': total_sents,
              'gold_state_seq': gold_state_seq,
              'gold_loc_seq': gold_loc_seq,
              'state_rel_labels': state_rel_labels,
              'loc_rel_labels': loc_rel_labels,
              'cpnet': cpnet_triples
              }
    return result


def write_output(output: List[Dict], output_filepath: str, sentences: List[List[str]], cpnet_cands: int):
    """
    Reads the headers of prediction file from dummy_filepath and fill in the blanks with prediction.
    Prediction will be stored according to output_filepath.
    """
    output_file = open(output_filepath, 'w', encoding='utf-8')
    columns = ['para_id', 'timestep', 'entity','gold_state', 'gold_location', 'sentence']
    columns += [f'cpnet_{idx}' for idx in range(1, cpnet_cands + 1)]
    output_file.write('\t'.join(columns) + '\n\n')
    assert len(sentences) == len(output)
    total_instances = len(output)

    for i in range(total_instances):
        instance = output[i]
        cpnet_triples = instance['cpnet']
        cpnet_line = ['' for _ in range(6)] + cpnet_triples
        output_file.write('\t'.join(cpnet_line) + '\n')
        sentence_list = sentences[i]
        sentence_list.insert(0, 'N/A')

        para_id = instance['id']
        entity_name = instance['entity']
        total_sents = instance['total_sents']
        state_rel_labels = instance['state_rel_labels']
        loc_rel_labels = instance['loc_rel_labels']
        gold_state_seq = instance['gold_state_seq']
        gold_state_seq.insert(0, 'N/A')
        gold_loc_seq = instance['gold_loc_seq']

        for step_i in range(total_sents + 1):  # number of states: total_sents + 1
            gold_state = gold_state_seq[step_i]
            gold_loc = gold_loc_seq[step_i]

            fields = [str(para_id), str(step_i), entity_name, gold_state, gold_loc, sentence_list[step_i]]
            if step_i > 0:
                fields += [f'{state_label} / {loc_label}' for state_label, loc_label in
                           zip(state_rel_labels[step_i-1], loc_rel_labels[step_i])]  # shift right to avoid location 0

            output_file.write('\t'.join(fields) + '\n')

    output_file.close()


def main():
    plm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = ProparaDataset(opt.dataset, opt=opt, tokenizer=plm_tokenizer, is_test=True)
    data_batch = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=Collate())

    output_result = []
    all_sentences = []
    cpnet_cands = 0
    all_labels = []

    for batch in tqdm(data_batch):
        gold_state_seq = batch['gold_state_seq']
        cpnet_triples = batch['cpnet']
        metadata = batch['metadata']
        state_rel_labels = batch['state_rel_labels']
        loc_rel_labels = batch['loc_rel_labels']
        all_sentences.extend(batch['sentences'])
        cpnet_cands = len(cpnet_triples[0])
        all_labels.append(state_rel_labels + loc_rel_labels[:, 1:])  # get rid of location 0

        batch_size = len(cpnet_triples)
        for i in range(batch_size):
            pred_instance = get_output(metadata=metadata[i],
                                       gold_state_seq=gold_state_seq[i].tolist(),
                                       cpnet_triples=cpnet_triples[i],
                                       state_rel_labels=state_rel_labels[i].tolist(),
                                       loc_rel_labels=loc_rel_labels[i].tolist())
            output_result.append(pred_instance)

    write_output(output=output_result, output_filepath=opt.output,
                 sentences = all_sentences, cpnet_cands = cpnet_cands)

    # TODO: count the labeled ratio by timestep (use total_sents)
    all_labels = torch.cat(all_labels, dim=0)
    labels_per_ins = all_labels.sum(dim=-1).sum(dim=-1)
    print(f'Number of instances that has at least one labeled ConceptNet triple: '
          f'{(labels_per_ins > 0).sum()}/{labels_per_ins.size(0)} '
          f'({(labels_per_ins > 0).sum().item() / labels_per_ins.size(0) * 100:.2f}%)')
    labels_per_triple = all_labels.sum(dim=-2)
    print(f'Number of sentences (timesteps) that has at least one labeled ConceptNet triple: '
          f'{(labels_per_triple > 0).sum()}/{labels_per_triple.size(0) * labels_per_triple.size(1)} '
          f'({(labels_per_triple > 0).sum().item() / (labels_per_triple.size(0) * labels_per_triple.size(1)) * 100:.2f}%)')


if __name__ == "__main__":
    main()