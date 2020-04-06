'''
 @Date  : 12/19/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

from typing import Dict, List
from Constants import *
from utils import *


def write_output(output: Dict[str, Dict], dummy_filepath: str, output_filepath: str):
    """
    Reads the headers of prediction file from dummy_filepath and fill in the blanks with prediction.
    Prediction will be stored according to output_filepath.
    """
    dummy_file = open(dummy_filepath, 'r', encoding='utf-8')
    output_file = open(output_filepath, 'w', encoding='utf-8')

    while True:

        dummy_line = dummy_file.readline()
        if not dummy_line:  # encouter EOF
            break

        fields = dummy_line.strip().split('\t')  # para_id, sent_id, entity, state (initially, NONE)
        assert len(fields) == 4 and fields[-1] == 'NONE'

        para_id = int(fields[0])
        sent_id = int(fields[1])
        entity_name = fields[2]
        pred_instance = output[str(para_id) + '-' + entity_name]

        total_sents = pred_instance['total_sents']
        assert sent_id <= total_sents
        assert para_id == pred_instance['id'] and entity_name == pred_instance['entity']

        prediction = pred_instance['prediction'][sent_id - 1]  # sent_id begins from 1
        state, loc_before, loc_after = prediction
        fields[-1] = state
        fields.append(loc_before)
        fields.append(loc_after)
        assert len(fields) == 6

        output_file.write('\t'.join(fields) + '\n')

    output_file.close()


def get_output(metadata: Dict, pred_state_seq: List[int], pred_loc_seq: List[int]) -> Dict:
    """
    Get the predicted output from generated sequences by the model.
    """
    para_id = metadata['para_id']
    entity_name = metadata['entity']
    loc_cand_list = metadata['loc_cand_list']
    total_sents = metadata['total_sents']

    pred_state_seq = [idx2state[idx] for idx in pred_state_seq]  # pred_state_seq outside the function won't be changed
    pred_loc_seq = [loc_cand_list[idx] for idx in pred_loc_seq]  # pred_loc_seq outside the function won't be changed

    pred_state_seq, pred_loc_seq = predict_consistent_loc(pred_state_seq = pred_state_seq, pred_loc_seq = pred_loc_seq,
                                                          para_id = para_id, entity = entity_name)
    prediction = format_final_prediction(pred_state_seq = pred_state_seq, pred_loc_seq = pred_loc_seq)
    assert len(prediction) == total_sents

    result = {'id': para_id,
              'entity': entity_name,
              'total_sents': total_sents,
              'prediction': prediction
              }
    return result


def format_final_prediction(pred_state_seq: List[str], pred_loc_seq: List[str]) -> List:
    """
    Final format: (state, loc_before, location_after) for each timestep (each sentence)
    """
    assert len(pred_state_seq) + 1 == len(pred_loc_seq)
    num_sents = len(pred_state_seq)
    prediction = []
    tag2state = {'O_C': 'NONE', 'O_D': 'NONE', 'C': 'CREATE', 'E': 'NONE', 'M': 'MOVE', 'D': 'DESTROY'}

    for i in range(num_sents):
        state_tag = tag2state[pred_state_seq[i]]
        prediction.append( hard_constraint(state_tag, pred_loc_seq[i], pred_loc_seq[i+1]) )

    return prediction


# TODO: if state == 'MOVE' but loc_before is identical to loc_after, there are two sets of solutions:
#       first, you can change its state to 'NONE'. Doing this will increase the precision of "moves" but will sacrifice its recall.
#       second, you can do nothing, which will increase the recall of "moves" but will sacrifice its precision.
def hard_constraint(state: str, loc_before: str, loc_after: str) -> (str, str, str):
    """
    Some commonsense hard constraints on the predictions.
    P.S. These constraints are only defined for evaluation, not for state sequence prediction.
    1. For state NONE, loc_after must be the same with loc_before
    2. For state MOVE and DESTROY, loc_before must not be '-'.
    3. For state CREATE, loc_before should be '-'.
    """
    if state == 'NONE' and loc_before != loc_after:
        if loc_after == '-':
            state = 'DESTROY'
        else:
            print('WHAT THE HELL?')
        # no other possibility
    if state == 'MOVE' and loc_before == '-':
        state = 'CREATE'
    # if state == 'MOVE' and loc_before == loc_after:
    #     state = 'NONE'
    if state == 'DESTROY' and loc_before == '-':
        state = 'NONE'
    if state == 'CREATE' and loc_before != '-':
        if loc_before == loc_after:
            state = 'NONE'
        elif loc_before != loc_after:
            state = 'MOVE'
    return state, loc_before, loc_after


# TODO: if state1 == 'E', then state0 should be '?' or state0 should be the same with state1?
def predict_consistent_loc(pred_state_seq: List[str], pred_loc_seq: List[str],
                           para_id: int, entity: str) -> (List[str], List[str]):
    """
    1. Only keep the location predictions at state "C" or "M"
    2. For "O_C", "O_D", and "D", location should be "-"
    3. For "E", location should be the same with previous timestep
    4. For state0: if state1 is "E", "M" or "D", then state0 should be "?";
       if state1 is "O_C", "O_D" or "C", then state0 should be "-"
    """

    assert len(pred_state_seq) == len(pred_loc_seq) - 1
    num_sents = len(pred_state_seq)
    consist_state_seq = []
    consist_loc_seq = []

    for sent_i in range(num_sents):

        state = pred_state_seq[sent_i]
        location = pred_loc_seq[sent_i + 1]

        #if 'D' is followed by a 'D'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'D' and state == 'D':
            state = 'E'

        # if 'D' is followed by a 'M'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'M' and state == 'D':
            state = 'E'

        # if 'E' is followed by a 'O_D'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'O_D' and state in ['E', 'M']:
            state = 'D'

        # if 'C' is followed by a 'C'
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'C' and state == 'C':
            state = 'O_C'

        # if the state before O_C is not O_C
        if sent_i < num_sents - 1 and pred_state_seq[sent_i + 1] == 'O_C' and state != 'O_C':
            temp_idx = sent_i + 1
            while temp_idx != num_sents and pred_state_seq[temp_idx] in ['O_C', 'O_D']:
                temp_idx += 1
            # pred_state_seq[temp_idx]: first state after O_C
            # state: last state before O_C
            if temp_idx != num_sents and pred_state_seq[temp_idx] == 'C':
                for idx in range(0, sent_i):
                    consist_state_seq[idx] = 'O_C'
                    consist_loc_seq[idx] = '-'
                state = 'O_C'
                if sent_i > 0:
                    consist_loc_seq[sent_i] = '-'
            else:
                for idx in range(sent_i+1, temp_idx):
                    pred_state_seq[idx] = 'E'

        # if 'O_C' is followed by a 'E'
        if sent_i > 0 and consist_state_seq[sent_i - 1] == 'O_C' and state == 'E':
            temp_idx = sent_i + 1
            while temp_idx != num_sents and pred_state_seq[temp_idx] == 'E':
                temp_idx += 1

            if temp_idx != num_sents and pred_state_seq[temp_idx] == 'C':
                for idx in range(sent_i, temp_idx):
                    pred_state_seq[idx] = 'O_C'
                state = 'O_C'
            else:
                state = 'C'

        # if 'O_C' is followed by a 'D'
        if sent_i > 0 and consist_state_seq[sent_i - 1] == 'O_C' and state == 'D' :
            for idx in range(0, sent_i):
                if pred_state_seq[idx] != 'O_C':
                    raise ValueError
                else:
                    consist_state_seq[idx] = 'E'
                    consist_loc_seq[idx] = pred_loc_seq[0]
            consist_loc_seq[sent_i] = pred_loc_seq[0]

        # set location according to state
        if sent_i == 0:
            location_0 = predict_loc0(state1 = state, loc0 = pred_loc_seq[0])
            consist_loc_seq.append(location_0)

        if state in ['O_C', 'O_D', 'D']:
            cur_location = '-'
        elif state == 'E':
            cur_location = consist_loc_seq[sent_i]  # this is the previous location since we add a location_0
        elif state in ['C', 'M']:
            cur_location = location

        consist_state_seq.append(state)
        consist_loc_seq.append(cur_location)

    assert len(consist_loc_seq) - 1 == len(consist_state_seq) == num_sents
    return consist_state_seq, consist_loc_seq


def predict_loc0(state1: str, loc0: str) -> str:

    assert state1 in state2idx.keys()

    if state1 in ['E', 'M', 'D']:
        loc0 = loc0
    elif state1 in ['O_C', 'O_D', 'C']:
        loc0 = '-'

    return loc0

# metadata = {'para_id': 249, 'entity': 'rocks ; smaller pieces', 'total_sents': 7, 'total_loc_cands': 4,
#             'loc_cand_list': ['pressure', 'pressure air', 'air', 'river', 'flower', 'water']}
# pred_state_seq = [4, 2, 2, 3, 5, 1, 1]
# pred_loc_seq = [0, 1, 2, 2, 4, 5, 3]
# print(get_output(metadata, pred_state_seq, pred_loc_seq))