'''
 @Date  : 04/03/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
 This script transforms a prediction.tsv file to the input format of evalQA.py
'''

import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-answer', type=str, default='../data/answers.tsv', help='answer.tsv file in leaderboard task')
    parser.add_argument('-output', type=str, default='all-moves.mod.tsv')
    opt = parser.parse_args()

    answer_file = open(opt.answer, 'r', encoding='utf8')
    output_file = open(opt.output, 'w', encoding='utf8')
    prev_pid = 37  # first instance in test set

    for line in answer_file:
        fields = line.strip().split('\t')
        assert len(fields) == 6
        pid, sid, entity, state, loc_before, loc_after = fields

        if int(pid) != prev_pid:
            output_file.write('\n')
            prev_pid = int(pid)

        if state == 'NONE':
            continue

        out_line = '\t'.join([pid, sid, '-', entity.lower(), state.lower(), loc_before.lower(), loc_after.lower()])
        output_file.write(out_line + '\n')

    output_file.write('\n')
    output_file.close()
    answer_file.close()


if __name__ == "__main__":
    main()
