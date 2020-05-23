'''
 @Date  : 12/26/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import time
print('[INFO] Starting import...')
import_start_time = time.time()
from torch.utils.data import DataLoader
from allennlp.modules.elmo import batch_to_ids
from Dataset import *
from Model import *
from predict import predict_loc0, predict_consistent_loc
import os
import re
import pdb
import argparse
print(f'[INFO] Import modules time: {time.time() - import_start_time}s')
torch.set_printoptions(precision=3, edgeitems=6, sci_mode=False)

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-plm_model_class', type=str, default='bert', help='pre-trained language model class')
parser.add_argument('-plm_model_name', type=str, default='bert-base-uncased', help='pre-trained language model name')
parser.add_argument('-hidden_size', type=int, default=256, help="hidden size of lstm")
parser.add_argument('-dropout', type=float, default=0, help="dropout rate")

parser.add_argument('-cpnet_path', type=str, default="ConceptNet/result/retrieval.json", help="path to conceptnet triples")
parser.add_argument('-cpnet_plm_path', type=str, default=None, help='specify to use pre-finetuned language model')
parser.add_argument('-cpnet_struc_input', action='store_true', default=False,
                    help='specify to use structural input format for ConceptNet triples')
parser.add_argument('-state_verb', type=str, default='ConceptNet/result/state_verb_cut.json', help='path to state verb dict')
parser.add_argument('-cpnet_inject', choices=['state', 'location', 'both', 'none'], default='both',
                    help='where to inject ConceptNet commonsense')
parser.add_argument('-wiki_plm_path', type=str, default=None, help='specify to use pre-finetuned language model')
parser.add_argument('-no_wiki', action='store_true', default=False, help='specify to exclude wiki')


parser.add_argument('-restore', type=str, default=None, help="restoring model path")
parser.add_argument('-test_set', type=str, default="data/test.json", help="path to test set")
parser.add_argument('-output', type=str, default=None, help="path to store prediction outputs")
parser.add_argument('-no_cuda', action='store_true', default=False, help="if true, will only use cpu")
opt = parser.parse_args()

plm_model_class, plm_tokenizer_class, plm_config_class = MODEL_CLASSES[opt.plm_model_class]
plm_tokenizer = plm_tokenizer_class.from_pretrained(opt.plm_model_name)


def get_output(metadata: Dict, pred_state_seq: List[int], pred_loc_seq: List[int],
               gold_state_seq: List[int], gold_loc_seq: List[int], cpnet_triples: List[str]) -> Dict:
    """
    Get the predicted output from generated sequences by the model.
    """
    para_id = metadata['para_id']
    entity_name = metadata['entity']
    loc_cand_list = metadata['loc_cand_list']
    total_sents = metadata['total_sents']

    pred_state_seq = [idx2state[idx] for idx in pred_state_seq]
    gold_state_seq = [idx2state[idx] for idx in gold_state_seq if idx != PAD_STATE]
    pred_loc_seq = [loc_cand_list[idx] for idx in pred_loc_seq]
    raw_gold_loc = metadata['raw_gold_loc']  # gold locations in string form

    pred_state_seq, pred_loc_seq = predict_consistent_loc(pred_state_seq = pred_state_seq, pred_loc_seq = pred_loc_seq,
                                                          para_id = para_id, entity = entity_name)
    assert len(pred_state_seq) == len(gold_state_seq) == len(pred_loc_seq) - 1 == len(raw_gold_loc) - 1 == total_sents

    prediction = []
    prediction.append( ('N/A', 'N/A', pred_loc_seq[0], raw_gold_loc[0]) )
    for i in range(total_sents):
        prediction.append( (pred_state_seq[i], gold_state_seq[i], pred_loc_seq[i+1], raw_gold_loc[i+1]) )

    result = {'id': para_id,
              'entity': entity_name,
              'total_sents': total_sents,
              'loc_cands': loc_cand_list,
              'prediction': prediction,
              'gold_loc_seq': gold_loc_seq,
              'cpnet': cpnet_triples
              }
    return result


def write_output(output: List[Dict], output_filepath: str, sentences: List[List[str]],
                 state_attn_log: List, loc_attn_log: List, cpnet_cands: int):
    """
    Reads the headers of prediction file from dummy_filepath and fill in the blanks with prediction.
    Prediction will be stored according to output_filepath.
    """
    print_attn = False
    if state_attn_log and loc_attn_log:
        print_attn = True

    output_file = open(output_filepath, 'w', encoding='utf-8')
    columns = ['para_id', 'timestep', 'entity', 'state', 'gold_state', 'location', 'gold_location', 'sentence']
    if print_attn:
        columns += [f'cpnet_{idx}' for idx in range(1, cpnet_cands + 1)]
    output_file.write('\t'.join(columns) + '\n\n')
    assert len(sentences) == len(output)
    total_instances = len(output)

    total_correct_state = 0
    total_correct_loc = 0
    total_predictions = 0

    for i in range(total_instances):
        instance = output[i]
        loc_cands = instance['loc_cands']
        if print_attn:
            cpnet_triples = instance['cpnet']
            cpnet_line = ['' for _ in range(8)] + cpnet_triples
            output_file.write('\t'.join(cpnet_line) + '\n')
        else:
            output_file.write('\n')
        sentence_list = sentences[i]
        sentence_list.insert(0, 'N/A')

        para_id = instance['id']
        entity_name = instance['entity']
        total_sents = instance['total_sents']
        gold_loc_seq = instance['gold_loc_seq']
        if print_attn:
            state_attn_list = state_attn_log[i]
            loc_attn_list = loc_attn_log[i]

        correct_state, correct_loc = 0, 0

        for step_i in range(total_sents + 1):  # number of states: total_sents + 1
            pred_state, gold_state, pred_loc, gold_loc = instance['prediction'][step_i]

            fields = [str(para_id), str(step_i), entity_name, pred_state, gold_state, pred_loc, gold_loc, sentence_list[step_i]]
            if step_i > 0 and print_attn:
                fields += [f'{state_attn:.2f}/{loc_attn:.2f}' if gold_loc_seq[step_i] >= 0 else f'{state_attn:.2f}/-'
                           for state_attn, loc_attn in
                           zip(state_attn_list[step_i-1], loc_attn_list[gold_loc_seq[step_i]][step_i])]

            if pred_state == gold_state and step_i > 0:
                correct_state += 1
            if pred_loc == gold_loc:
                correct_loc += 1

            output_file.write('\t'.join(fields) + '\n')

        total_correct_state += correct_state
        total_correct_loc += correct_loc
        total_predictions += total_sents

        state_accuracy = correct_state / total_sents
        loc_accuracy = correct_loc / (total_sents + 1)
        footer = [str(para_id), '', entity_name, f'{correct_state}/{total_sents}', f'{state_accuracy*100:.1f}%',
                  f'{correct_loc}/{total_sents+1}', f'{loc_accuracy*100:.1f}%', '']
        output_file.write('\t'.join(footer) + '\n\n')

    output_file.close()

    total_accuracy = (total_correct_state + total_correct_loc) / (2 * total_predictions + total_instances)
    state_accuracy = total_correct_state / total_predictions
    loc_accuracy = total_correct_loc / (total_predictions + total_instances)

    print(f'Final Prediction:\n'
          f'Total Accuracy: {total_accuracy * 100:.3f}%, '
          f'State Prediction Accuracy: {state_accuracy * 100:.3f}%, '
          f'Location Accuracy: {loc_accuracy * 100:.3f}%')


def test(test_set, model):
    print('[INFO] Start testing...')
    test_batch = DataLoader(dataset = test_set, batch_size = opt.batch_size, shuffle = False, collate_fn = Collate())

    start_time = time.time()
    report_state_correct, report_state_pred = 0, 0
    report_loc_correct, report_loc_pred = 0, 0
    cpnet_cands = 0
    output_result = []
    all_sentences = []

    with torch.no_grad():
        for batch in test_batch:

            paragraphs = batch['paragraph']
            token_ids = plm_tokenizer.batch_encode_plus(paragraphs, add_special_tokens=True,
                                                        return_tensors='pt')['input_ids']
            all_sentences.extend(batch['sentences'])
            sentence_mask = batch['sentence_mask']
            entity_mask = batch['entity_mask']
            verb_mask = batch['verb_mask']
            loc_mask = batch['loc_mask']
            gold_loc_seq = batch['gold_loc_seq']
            gold_state_seq = batch['gold_state_seq']
            cpnet_triples = batch['cpnet']
            state_rel_labels = batch['state_rel_labels']
            loc_rel_labels = batch['loc_rel_labels']
            metadata = batch['metadata']
            cpnet_cands = len(cpnet_triples[0])
            num_cands = torch.IntTensor([meta['total_loc_cands'] + 1 for meta in metadata])  # +1 for unk

            if not opt.no_cuda:
                token_ids = token_ids.cuda()
                sentence_mask = sentence_mask.cuda()
                entity_mask = entity_mask.cuda()
                verb_mask = verb_mask.cuda()
                loc_mask = loc_mask.cuda()
                gold_loc_seq = gold_loc_seq.cuda()
                gold_state_seq = gold_state_seq.cuda()
                state_rel_labels = state_rel_labels.cuda()
                loc_rel_labels = loc_rel_labels.cuda()
                num_cands = num_cands.cuda()

            test_result = model(token_ids=token_ids, entity_mask=entity_mask, verb_mask=verb_mask,
                                loc_mask=loc_mask, gold_loc_seq=gold_loc_seq, gold_state_seq=gold_state_seq,
                                num_cands=num_cands, sentence_mask=sentence_mask, cpnet_triples=cpnet_triples,
                                state_rel_labels=state_rel_labels, loc_rel_labels=loc_rel_labels)

            pred_state_seq, pred_loc_seq, test_state_correct, test_state_pred,\
                test_loc_correct, test_loc_pred = test_result

            batch_size = len(paragraphs)
            for i in range(batch_size):
                pred_instance = get_output(metadata = metadata[i],
                                           pred_state_seq = pred_state_seq[i],
                                           pred_loc_seq = pred_loc_seq[i],
                                           gold_state_seq = gold_state_seq[i].tolist(),
                                           gold_loc_seq = gold_loc_seq[i].tolist(),
                                           cpnet_triples = cpnet_triples[i])
                output_result.append(pred_instance)

            report_state_correct += test_state_correct
            report_state_pred += test_state_pred
            report_loc_correct += test_loc_correct
            report_loc_pred += test_loc_pred

    total_accuracy = (report_state_correct + report_loc_correct) / (report_state_pred + report_loc_pred)
    state_accuracy = report_state_correct / report_state_pred
    loc_accuracy = report_loc_correct / report_loc_pred

    print(f'Test:\n'
           f'Total Accuracy: {total_accuracy * 100:.3f}%, '
           f'State Prediction Accuracy: {state_accuracy * 100:.3f}%, '
           f'Location Accuracy: {loc_accuracy * 100:.3f}%')

    state_attn_log = model.StateTracker.CpnetMemory.AttnUpdate.attn_log
    loc_attn_log = model.LocationPredictor.CpnetMemory.AttnUpdate.attn_log

    write_output(output = output_result, output_filepath = opt.output, sentences = all_sentences,
                 state_attn_log = state_attn_log, loc_attn_log = loc_attn_log, cpnet_cands = cpnet_cands)
    print(f'[INFO] Test finished. Time elapse: {time.time() - start_time}s')


if __name__ == "__main__":
    test_set = ProparaDataset(opt.test_set, opt=opt, tokenizer=plm_tokenizer, is_test=True)

    print('[INFO] Start loading trained model...')
    restore_start_time = time.time()
    model = KOALA(opt=opt, is_test=True)
    model_state_dict = torch.load(opt.restore)
    model.load_state_dict(model_state_dict)
    model.eval()
    print(f'[INFO] Loaded model from {opt.restore}, time elapse: {time.time() - restore_start_time}s')

    if not opt.no_cuda:
        model.cuda()
    test(test_set, model)