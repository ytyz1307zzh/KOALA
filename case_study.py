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
parser.add_argument('-hidden_size', type=int, default=128, help="hidden size of lstm")
parser.add_argument('-lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('-dropout', type=float, default=0.5, help="dropout rate")
parser.add_argument('-loc_loss', type=float, default=0.3, help="hyper-parameter to weight location loss and state_loss")
parser.add_argument('-elmo_dir', type=str, default='elmo', help="directory that contains options and weight files for allennlp Elmo")

parser.add_argument('-cpnet', type=str, default="ConceptNet/result/retrieval.json", help="path to conceptnet triples")
parser.add_argument('-wiki', type=str, default="wiki/result/retrieval.json", help="path to wiki paragraphs")
parser.add_argument('-cpnet_inject', choices=['state', 'location', 'both'], default='both',
                    help='where to inject ConceptNet commonsense')

parser.add_argument('-restore', type=str, default=None, help="restoring model path")
parser.add_argument('-test_set', type=str, default="data/test.json", help="path to test set")
parser.add_argument('-output', type=str, default=None, help="path to store prediction outputs")
parser.add_argument('-no_cuda', action='store_true', default=False, help="if true, will only use cpu")
opt = parser.parse_args()

plm_model_class, plm_tokenizer_class, plm_config_class = MODEL_CLASSES[opt.plm_model_class]
plm_tokenizer = plm_tokenizer_class.from_pretrained(opt.plm_model_name)


def predict_loc0(state1: str) -> str:

    assert state1 in state2idx.keys()

    if state1 in ['E', 'M', 'D']:
        loc0 = '?'
    elif state1 in ['O_C', 'O_D', 'C']:
        loc0 = '-'

    return loc0


def predict_consistent_loc(pred_state_seq: List[str], pred_loc_seq: List[str]) -> List[str]:
    """
    1. Only keep the location predictions at state "C" or "M"
    2. For "O_C", "O_D", and "D", location should be "-"
    3. For "E", location should be the same with previous timestep
    4. For state0: if state1 is "E", "M" or "D", then state0 should be "?";
       if state1 is "O_C", "O_D" or "C", then state0 should be "-"
    """

    assert len(pred_state_seq) == len(pred_loc_seq)
    num_sents = len(pred_state_seq)
    consist_loc_seq = []

    for sent_i in range(num_sents):

        state = pred_state_seq[sent_i]
        location = pred_loc_seq[sent_i]

        if sent_i == 0:
            location_0 = predict_loc0(state1 = state)
            consist_loc_seq.append(location_0)

        if state in ['O_C', 'O_D', 'D']:
            cur_location = '-'
        elif state == 'E':
            cur_location = consist_loc_seq[sent_i]  # this is the previous location since we add a location_0
        elif state in ['C', 'M']:
            cur_location = location

        consist_loc_seq.append(cur_location)

    assert len(consist_loc_seq) == num_sents + 1
    return consist_loc_seq


def get_output(metadata: Dict, pred_state_seq: List[int], pred_loc_seq: List[int],
               gold_state_seq: List[int], cpnet_triples: List[str]) -> Dict:
    """
    Get the predicted output from generated sequences by the model.
    """
    para_id = metadata['para_id']
    entity_name = metadata['entity']
    loc_cand_list = metadata['loc_cand_list']
    total_sents = metadata['total_sents']
    loc_cands = metadata['loc_cand_list']

    pred_state_seq = [idx2state[idx] for idx in pred_state_seq]
    gold_state_seq = [idx2state[idx] for idx in gold_state_seq if idx != PAD_STATE]
    pred_loc_seq = [loc_cand_list[idx] for idx in pred_loc_seq]
    gold_loc_seq = metadata['raw_gold_loc']  # gold locations in string form

    pred_loc_seq = predict_consistent_loc(pred_state_seq = pred_state_seq, pred_loc_seq = pred_loc_seq)
    assert len(pred_state_seq) == len(gold_state_seq) == len(pred_loc_seq) - 1 == len(gold_loc_seq) - 1 == total_sents

    prediction = []
    prediction.append( ('N/A', 'N/A', pred_loc_seq[0], gold_loc_seq[0]) )
    for i in range(total_sents):
        prediction.append( (pred_state_seq[i], gold_state_seq[i], pred_loc_seq[i+1], gold_loc_seq[i+1]) )

    result = {'id': para_id,
              'entity': entity_name,
              'total_sents': total_sents,
              'loc_cands': loc_cand_list,
              'prediction': prediction,
              'cpnet': cpnet_triples
              }
    return result


def write_output(output: List[Dict], output_filepath: str, sentences: List[List[str]],
                 state_attn_log: List, loc_attn_log: List, cpnet_cands: int):
    """
    Reads the headers of prediction file from dummy_filepath and fill in the blanks with prediction.
    Prediction will be stored according to output_filepath.
    """
    output_file = open(output_filepath, 'w', encoding='utf-8')
    columns = ['para_id', 'timestep', 'entity', 'state', 'gold_state', 'location', 'gold_location', 'sentence']
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
        assert len(loc_cands) > 3
        cpnet_triples = instance['cpnet']
        loc_cand_line = [f'loc: {loc_cands[0]}/{loc_cands[1]}/{loc_cands[2]}']
        cpnet_line = ['' for _ in range(7)] + loc_cand_line + cpnet_triples
        output_file.write('\t'.join(cpnet_line) + '\n')
        sentence_list = sentences[i]
        sentence_list.insert(0, 'N/A')

        para_id = instance['id']
        entity_name = instance['entity']
        total_sents = instance['total_sents']
        state_attn_list = state_attn_log[i]
        loc_attn_list = loc_attn_log[i][:3]

        correct_state, correct_loc = 0, 0

        for step_i in range(total_sents + 1):  # number of states: total_sents + 1
            pred_state, gold_state, pred_loc, gold_loc = instance['prediction'][step_i]

            fields = [str(para_id), str(step_i), entity_name, pred_state, gold_state, pred_loc, gold_loc, sentence_list[step_i]]
            if step_i > 0:
                fields += [f'{state_attn:.2f}/{loc0_attn:.2f}/{loc1_attn:.2f}/{loc2_attn:.2f}'
                           for state_attn, loc0_attn, loc1_attn, loc2_attn in
                           zip(state_attn_list[step_i-1], loc_attn_list[0][step_i-1],
                               loc_attn_list[1][step_i - 1], loc_attn_list[2][step_i-1])]

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
            enc_outputs = plm_tokenizer.batch_encode_plus(paragraphs, add_special_tokens=True, return_tensors='pt')
            token_ids = enc_outputs['input_ids']
            all_sentences.extend(batch['sentences'])
            sentence_mask = batch['sentence_mask']
            entity_mask = batch['entity_mask']
            verb_mask = batch['verb_mask']
            loc_mask = batch['loc_mask']
            gold_loc_seq = batch['gold_loc_seq']
            gold_state_seq = batch['gold_state_seq']
            cpnet_triples = batch['cpnet']
            metadata = batch['metadata']
            cpnet_cands = len(cpnet_triples[0])
            num_cands = torch.IntTensor([meta['total_loc_cands'] for meta in metadata])

            if not opt.no_cuda:
                token_ids = token_ids.cuda()
                sentence_mask = sentence_mask.cuda()
                entity_mask = entity_mask.cuda()
                verb_mask = verb_mask.cuda()
                loc_mask = loc_mask.cuda()
                gold_loc_seq = gold_loc_seq.cuda()
                gold_state_seq = gold_state_seq.cuda()
                num_cands = num_cands.cuda()

            test_result = model(token_ids=token_ids, entity_mask=entity_mask, verb_mask=verb_mask,
                                loc_mask=loc_mask, gold_loc_seq=gold_loc_seq, gold_state_seq=gold_state_seq,
                                num_cands=num_cands, sentence_mask=sentence_mask, cpnet_triples=cpnet_triples,
                                print_hidden=False)

            pred_state_seq, pred_loc_seq, test_state_correct, test_state_pred,\
                test_loc_correct, test_loc_pred = test_result

            batch_size = len(paragraphs)
            for i in range(batch_size):
                pred_instance = get_output(metadata = metadata[i],
                                           pred_state_seq = pred_state_seq[i],
                                           pred_loc_seq = pred_loc_seq[i],
                                           gold_state_seq = gold_state_seq[i].tolist(),
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
    test_set = ProparaDataset(opt.test_set, cpnet_path=opt.cpnet, tokenizer=plm_tokenizer, is_test=True)

    print('[INFO] Start loading trained model...')
    restore_start_time = time.time()
    model = NCETModel(opt=opt, is_test=True)
    model_state_dict = torch.load(opt.restore)
    model.load_state_dict(model_state_dict)
    model.eval()
    print(f'[INFO] Loaded model from {opt.restore}, time elapse: {time.time() - restore_start_time}s')

    if not opt.no_cuda:
        model.cuda()
    test(test_set, model)