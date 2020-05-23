'''
 @Date  : 12/11/2019
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''
import time
print('[INFO] Starting import...')
import_start_time = time.time()
import torch
import json
import os
import pdb
import random
import numpy as np
from typing import List, Dict
from Constants import *
import argparse
# from torchsummaryX import summary
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import *
from predict import *
from Dataset import *
from Model import *
print(f'[INFO] Import modules time: {time.time() - import_start_time}s')
torch.set_printoptions(precision=3, edgeitems=6, sci_mode=False, threshold=3000)


parser = argparse.ArgumentParser()

# model parameters
parser.add_argument('-per_gpu_batch_size', type=int, default=32)
parser.add_argument('-plm_model_class', type=str, default='bert', help='pre-trained language model class')
parser.add_argument('-plm_model_name', type=str, default='bert-base-uncased', help='pre-trained language model name')
parser.add_argument('-hidden_size', type=int, default=256, help="hidden size of lstm")
parser.add_argument('-lr', type=float, default=3e-5, help="learning rate")
parser.add_argument('-dropout', type=float, default=0.4, help="dropout rate")
parser.add_argument('-loc_loss', type=float, default=0.3, help="hyper-parameter to weight location loss")
parser.add_argument('-attn_loss', type=float, default=0.5, help="hyper-parameter to weight attention loss")
parser.add_argument('-max_grad_norm', default=1.0, type=float, help="Max gradient norm")
parser.add_argument('-grad_accum_step', default=1, type=int, help='gradient accumulation steps')

# training parameters
parser.add_argument('-mode', type=str, choices=['train', 'test'], default='train', help="train or test")
parser.add_argument('-ckpt_dir', type=str, default=None, help="directory to save checkpoints and logs")
parser.add_argument('-save_mode', type=str, choices=['best', 'all', 'none', 'last', 'best-last'], default='best',
                    help="best (default): save checkpoints when reaching new best score; all: save all checkpoints; "
                         "none: don't save; best-last: save both the best and the last checkpoint")
parser.add_argument('-epoch', type=int, default=100, help="number of epochs, use -1 to rely on early stopping only")
parser.add_argument('-impatience', type=int, default=20,
                    help='number of evaluation rounds for early stopping, use -1 to disable early stopping')
parser.add_argument('-report', type=int, default=2, help="report frequence per epoch, should be at least 1")
parser.add_argument('-train_set', type=str, default="data/train.json", help="path to training set")
parser.add_argument('-dev_set', type=str, default="data/dev.json", help="path to dev set")
parser.add_argument('-no_cuda', action='store_true', default=False, help="if true, will only use cpu")

# test parameters
parser.add_argument('-test_set', type=str, default="data/test.json", help="path to test set")
parser.add_argument('-restore', type=str, default=None, help="path to saved checkpoint")
parser.add_argument('-dummy_test', type=str, default="data/dummy-predictions.tsv", help="path to prediction file template")
parser.add_argument('-output', type=str, default=None, help="path to store prediction outputs")

# commonsense parameters
parser.add_argument('-cpnet_path', type=str, default="ConceptNet/result/retrieval.json",
                    help="path to the retrieved ConceptNet knowledge triples")
parser.add_argument('-cpnet_plm_path', type=str, default=None,
                    help='specify to use pre-fine-tuned knowledge encoder on ConceptNet triples')
parser.add_argument('-cpnet_struc_input', action='store_true', default=False,
                    help='if true, use structural input format for ConceptNet triples')
parser.add_argument('-state_verb', type=str, default='ConceptNet/result/state_verb_cut.json',
                    help='path to co-appearance verb set of entity states')
parser.add_argument('-cpnet_inject', choices=['state', 'location', 'both', 'none'], default='both',
                    help='where to inject ConceptNet commonsense, select "none" to avoid infusing ConceptNet')
parser.add_argument('-wiki_plm_path', type=str, default=None,
                    help='specify to use pre-fine-tuned text encoder on Wiki paragraphs')
parser.add_argument('-finetune', action='store_true', default=False, help='if true, fine-tune the bert encoder')
parser.add_argument('-no_wiki', action='store_true', default=False, help='if true, use the vanilla PLM from huggingface')

opt = parser.parse_args()

try:
    opt.n_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
except KeyError:  # did not specify device from cmd
    opt.n_gpu = 1
opt.batch_size = opt.per_gpu_batch_size * opt.n_gpu

if opt.cpnet_inject == 'none':
    opt.attn_loss = 0

if opt.cpnet_struc_input:
    assert opt.cpnet_plm_path is not None

plm_model_class, plm_tokenizer_class, plm_config_class = MODEL_CLASSES[opt.plm_model_class]
plm_tokenizer = plm_tokenizer_class.from_pretrained(opt.plm_model_name)

if opt.ckpt_dir and not os.path.exists(opt.ckpt_dir):
    os.mkdir(opt.ckpt_dir)
# prepare logger
if opt.ckpt_dir:
    log_path = os.path.join(opt.ckpt_dir, 'train.log')
    if os.path.exists(log_path):
        log_file = open(log_path, 'a', encoding='utf-8')
    else:
        log_file = open(log_path, 'w', encoding='utf-8')


def output(text):
    print(text)
    if opt.ckpt_dir:
        print(text, file = log_file)


output('Received arguments:')
output(opt)
output('-' * 50)

assert opt.report >= 1

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
if opt.n_gpu > 0:
    torch.cuda.manual_seed_all(1234)


def save_model(ckpt_dir, model_name, model: nn.Module, optimizer):
    if opt.save_mode == 'none':
        return

    if not opt.ckpt_dir:
        print("[ERROR] Intended to store checkpoint but no checkpoint directory is specified.")
        raise RuntimeError("Did not specify -ckpt_dir option")

    model_to_save = model.module if hasattr(model, "module") else model
    model_state_dict = model_to_save.state_dict()
    torch.save(model_state_dict, os.path.join(ckpt_dir, model_name))

    if opt.save_mode in ['last', 'best-last', 'all']:
        optim_state_dict = optimizer.state_dict()
        torch.save(optim_state_dict, os.path.join(ckpt_dir, "optimizer.pt"))


def train():

    if opt.ckpt_dir:
        if opt.ckpt_dir.endswith('/'):
            tb_writer = SummaryWriter(logdir=os.path.join('runs', opt.ckpt_dir.split('/')[-2]))
        else:
            tb_writer = SummaryWriter(logdir=os.path.join('runs', opt.ckpt_dir.split('/')[-1]))
    else:
        tb_writer = SummaryWriter()

    train_set = ProparaDataset(opt.train_set, opt=opt, tokenizer=plm_tokenizer, is_test=False)

    train_batch = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True, collate_fn=Collate())
    dev_set = ProparaDataset(opt.dev_set, opt=opt, tokenizer=plm_tokenizer, is_test=False)

    model = KOALA(opt = opt, is_test = False)
    if not opt.no_cuda:
        model.cuda()
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    if opt.restore is not None:
        model_state_dict = torch.load(opt.restore)
        model.load_state_dict(model_state_dict)
        optim_state_dict = torch.load(os.path.join(opt.ckpt_dir, "optimizer.pt"))
        optimizer.load_state_dict(optim_state_dict)
        print(f'[INFO] Loaded model and optimizer from {opt.ckpt_dir}, resume training...')

    best_score = np.NINF
    impatience = 0
    epoch_i = 0
    report_cnt = 0

    if opt.epoch == -1:
        opt.epoch = np.inf

    if opt.impatience == -1:
        opt.impatience = np.inf

    print('Start training...')
    model.zero_grad()

    while epoch_i < opt.epoch:

        model.train()
        train_instances = len(train_set)

        start_time = time.time()
        report_state_loss, report_loc_loss = 0, 0
        report_state_correct, report_state_pred = 0, 0
        report_loc_correct, report_loc_pred = 0, 0
        report_attn_loss, report_attn_pred = 0, 0
        batch_cnt = 0

        if train_instances % opt.batch_size == 0:
            total_batches = train_instances // opt.batch_size
        else:
            total_batches = train_instances // opt.batch_size + 1
        report_batch = get_report_time(total_batches = total_batches,
                                       report_times = opt.report,
                                       grad_accum_step = opt.grad_accum_step)  # when to report results

        for batch in train_batch:

            paragraphs = batch['paragraph']
            token_ids = plm_tokenizer.batch_encode_plus(paragraphs, add_special_tokens=True,
                                                        return_tensors='pt')['input_ids']
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

            train_result = model(token_ids = token_ids, entity_mask = entity_mask, verb_mask = verb_mask,
                                 loc_mask = loc_mask, gold_loc_seq = gold_loc_seq, gold_state_seq = gold_state_seq,
                                 num_cands = num_cands, sentence_mask = sentence_mask, cpnet_triples = cpnet_triples,
                                 state_rel_labels = state_rel_labels, loc_rel_labels = loc_rel_labels)

            train_state_loss, train_loc_loss, train_attn_loss, train_state_correct,\
            train_state_pred, train_loc_correct, train_loc_pred, train_attn_pred = train_result

            if opt.n_gpu > 1:
                train_state_loss = train_state_loss.mean()
                train_loc_loss = train_loc_loss.mean()
                if train_attn_loss is not None:
                    train_attn_loss = train_attn_loss.mean()

            train_loss = train_state_loss + opt.loc_loss * train_loc_loss
            if train_attn_loss is not None:
                train_loss += opt.attn_loss * train_attn_loss

            if opt.grad_accum_step > 1:
                train_loss = train_loss / opt.grad_accum_step
            train_loss.backward()

            report_state_loss += train_state_loss.item() * train_state_pred
            report_loc_loss += train_loc_loss.item() * train_loc_pred
            report_state_correct += train_state_correct
            report_state_pred += train_state_pred
            report_loc_correct += train_loc_correct
            report_loc_pred += train_loc_pred
            if train_attn_loss is not None:
                report_attn_loss += train_attn_loss.item() * train_attn_pred
                report_attn_pred += train_attn_pred
            batch_cnt += 1

            # gradient accumulation
            if batch_cnt % opt.grad_accum_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optimizer.step()
                model.zero_grad()

                # time to report results
                if batch_cnt in report_batch:

                    state_loss = report_state_loss / report_state_pred  # average over all elements
                    loc_loss = report_loc_loss / report_loc_pred
                    total_loss = state_loss + opt.loc_loss * loc_loss

                    if train_attn_loss is not None:
                        attn_loss = report_attn_loss / report_attn_pred
                        total_loss += opt.attn_loss * attn_loss
                    else:
                        attn_loss = 0

                    state_accuracy = report_state_correct / report_state_pred
                    loc_accuracy = report_loc_correct / report_loc_pred
                    total_accuracy = (report_state_correct + report_loc_correct) / (report_state_pred + report_loc_pred)

                    output('*' * 50)
                    output(f'{batch_cnt}/{total_batches}, Epoch {epoch_i+1}:\n'
                           f'Loss: {total_loss:.3f}, State Loss: {state_loss:.3f}, '
                           f'Location Loss: {loc_loss:.3f}, Attention Loss: {attn_loss:.3f}\n'
                           f'Total Accuracy: {total_accuracy*100:.3f}%, '
                           f'State Prediction Accuracy: {state_accuracy*100:.3f}%, '
                           f'Location Accuracy: {loc_accuracy*100:.3f}% \n'
                           f'Time Elapse: {time.time()-start_time:.2f}s')
                    output('-' * 50)

                    report_cnt += 1
                    tb_writer.add_scalar('train_state_loss', state_loss, report_cnt)
                    tb_writer.add_scalar('train_loc_loss', loc_loss, report_cnt)

                    model.eval()
                    eval_score = evaluate(dev_set, model, tb_writer, report_cnt)
                    model.train()

                    if eval_score > best_score:  # new best score
                        best_score = eval_score
                        impatience = 0
                        output('New best score!')
                        if opt.save_mode == 'all':
                            save_model(opt.ckpt_dir, f'best_checkpoint_{best_score:.3f}.pt', model, optimizer)
                        elif opt.save_mode in ['best', 'best-last']:
                            save_model(opt.ckpt_dir, f'best_checkpoint.pt', model, optimizer)
                    else:
                        impatience += 1
                        output(f'Impatience: {impatience}, best score: {best_score:.3f}.')
                        if opt.save_mode == 'all':
                            save_model(opt.ckpt_dir, f'checkpoint{report_cnt}_{eval_score:.3f}.pt', model, optimizer)
                        if impatience >= opt.impatience:
                            output('Early Stopping!')
                            if opt.save_mode in ['last', 'best-last']:
                                save_model(opt.ckpt_dir, f'checkpoint{report_cnt}_{eval_score:.3f}.pt', model, optimizer)
                            tb_writer.close()
                            quit()

                    report_state_loss, report_loc_loss = 0, 0
                    report_state_correct, report_state_pred = 0, 0
                    report_loc_correct, report_loc_pred = 0, 0
                    report_attn_loss, report_attn_pred = 0, 0
                    start_time = time.time()

        epoch_i += 1

    if opt.save_mode in ['last', 'best-last']:
        save_model(opt.ckpt_dir, f'checkpoint{report_cnt}_{eval_score:.3f}.pt', model, optimizer)
    tb_writer.close()


        # summary(model, char_paragraph, entity_mask, verb_mask, loc_mask)
        # with SummaryWriter() as writer:
        #     writer.add_graph(model, (char_paragraph, entity_mask, verb_mask, loc_mask, gold_loc_mask, gold_state_mask))


def evaluate(dev_set, model, tb_writer, report_cnt: int):
    dev_batch = DataLoader(dataset=dev_set, batch_size=opt.batch_size, shuffle=False, collate_fn=Collate())

    start_time = time.time()
    report_state_loss, report_loc_loss = 0, 0
    report_state_correct, report_state_pred = 0, 0
    report_loc_correct, report_loc_pred = 0, 0
    report_attn_loss, report_attn_pred = 0, 0
    batch_cnt = 0

    with torch.no_grad():
        for batch in dev_batch:

            paragraphs = batch['paragraph']
            token_ids = plm_tokenizer.batch_encode_plus(paragraphs, add_special_tokens=True,
                                                        return_tensors='pt')['input_ids']
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

            eval_result = model(token_ids = token_ids, entity_mask = entity_mask, verb_mask = verb_mask,
                                loc_mask = loc_mask, gold_loc_seq = gold_loc_seq, gold_state_seq = gold_state_seq,
                                num_cands = num_cands, sentence_mask = sentence_mask, cpnet_triples = cpnet_triples,
                                state_rel_labels = state_rel_labels, loc_rel_labels = loc_rel_labels)

            eval_state_loss, eval_loc_loss, eval_attn_loss, eval_state_correct,\
            eval_state_pred, eval_loc_correct, eval_loc_pred, eval_attn_pred = eval_result

            if opt.n_gpu > 1:
                eval_state_loss = eval_state_loss.mean()
                eval_loc_loss = eval_loc_loss.mean()
                if eval_attn_loss is not None:
                    eval_attn_loss = eval_attn_loss.mean()

            report_state_loss += eval_state_loss.item() * eval_state_pred
            report_loc_loss += eval_loc_loss.item() * eval_loc_pred
            report_state_correct += eval_state_correct
            report_state_pred += eval_state_pred
            report_loc_correct += eval_loc_correct
            report_loc_pred += eval_loc_pred
            if eval_attn_loss is not None:
                report_attn_loss += eval_attn_loss.item() * eval_attn_pred
                report_attn_pred += eval_attn_pred

            batch_cnt += 1

    state_loss = report_state_loss / report_state_pred  # average over all elements
    loc_loss = report_loc_loss / report_loc_pred
    total_loss = state_loss + opt.loc_loss * loc_loss
    if eval_attn_loss is not None:
        attn_loss = report_attn_loss / report_attn_pred
        total_loss += opt.attn_loss * attn_loss
    else:
        attn_loss = 0

    total_accuracy = (report_state_correct + report_loc_correct) / (report_state_pred + report_loc_pred)
    state_accuracy = report_state_correct / report_state_pred
    loc_accuracy = report_loc_correct / report_loc_pred

    output(f'\tEvaluation:\n'
           f'\tLoss: {total_loss:.3f}, State Loss: {state_loss:.3f}, '
           f'Location Loss: {loc_loss:.3f}, Attention Loss: {attn_loss:.3f}\n'
           f'\tTotal Accuracy: {total_accuracy * 100:.3f}%, '
           f'State Prediction Accuracy: {state_accuracy * 100:.3f}%, '
           f'Location Accuracy: {loc_accuracy * 100:.3f}% \n'
           f'\tTime Elapse: {time.time() - start_time:.2f}s')
    output('*' * 50)

    tb_writer.add_scalar('eval_state_loss', state_loss, report_cnt)
    tb_writer.add_scalar('eval_loc_loss', loc_loss, report_cnt)

    return total_accuracy * 100


def test(test_set, model):

    print('[INFO] Start testing...')
    test_batch = DataLoader(dataset=test_set, batch_size=opt.batch_size, shuffle=False, collate_fn=Collate())

    start_time = time.time()
    report_state_correct, report_state_pred = 0, 0
    report_loc_correct, report_loc_pred = 0, 0
    output_result = {}
    batch_cnt = 0

    with torch.no_grad():
        for batch in test_batch:

            paragraphs = batch['paragraph']
            token_ids = plm_tokenizer.batch_encode_plus(paragraphs, add_special_tokens=True,
                                                        return_tensors='pt')['input_ids']

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
                pred_instance = get_output(metadata = metadata[i], pred_state_seq = pred_state_seq[i], pred_loc_seq = pred_loc_seq[i])
                para_id = pred_instance['id']
                entity_name = pred_instance['entity']
                output_result[str(para_id) + '-' + entity_name] = pred_instance

            report_state_correct += test_state_correct
            report_state_pred += test_state_pred
            report_loc_correct += test_loc_correct
            report_loc_pred += test_loc_pred

            batch_cnt += 1

    total_accuracy = (report_state_correct + report_loc_correct) / (report_state_pred + report_loc_pred)
    state_accuracy = report_state_correct / report_state_pred
    loc_accuracy = report_loc_correct / report_loc_pred

    output(f'Test:\n'
           f'Total Accuracy: {total_accuracy * 100:.3f}%, '
           f'State Prediction Accuracy: {state_accuracy * 100:.3f}%, '
           f'Location Accuracy: {loc_accuracy * 100:.3f}%')

    write_output(output = output_result, dummy_filepath = opt.dummy_test, output_filepath = opt.output)
    print(f'[INFO] Test finished. Time elapse: {time.time() - start_time}s')


if __name__ == "__main__":

    if opt.mode == 'train':
        train()

    elif opt.mode == 'test':
        if not opt.restore:
            print("[ERROR] Entered test mode but no restore file is specified.")
            raise RuntimeError("Did not specify -restore option")

        if not opt.output:
            print("[ERROR] Entered test mode but no output file is specified.")
            raise RuntimeError("Did not specify -output option")

        if not opt.output.endswith('.tsv'):
            print("[WARNING] The output will be in TSV format, while the specified output file does not have .tsv suffix.")

        plm_model_class, plm_tokenizer_class, plm_config_class = MODEL_CLASSES[opt.plm_model_class]
        plm_tokenizer = plm_tokenizer_class.from_pretrained(opt.plm_model_name)

        test_set = ProparaDataset(opt.test_set, opt=opt, tokenizer=plm_tokenizer, is_test=True)

        print('[INFO] Start loading trained model...')
        restore_start_time = time.time()
        model = KOALA(opt = opt, is_test = True)
        model_state_dict = torch.load(opt.restore)
        model.load_state_dict(model_state_dict)
        model.eval()
        print(f'[INFO] Loaded model from {opt.restore}, time elapse: {time.time() - restore_start_time}s')

        if not opt.no_cuda:
            model.cuda()

        test(test_set, model)