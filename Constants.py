# map state labels to indices
state2idx = {'O_C': 0, 'O_D': 1, 'E': 2, 'M': 3, 'C': 4, 'D': 5}
idx2state = {0: 'O_C', 1: 'O_D', 2: 'E', 3: 'M', 4: 'C', 5: 'D'}

UNK_LOC = -1
NIL_LOC = -2
PAD_LOC = -3

PAD_STATE = -1

NUM_STATES = len(state2idx)