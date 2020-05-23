'''
 @Date  : 01/03/2020
 @Author: Zhihan Zhang
 @mail  : zhangzhihan@pku.edu.cn
 @homepage: ytyz1307zzh.github.io
'''

import time
import os
import json
import argparse
import logging
from typing import Dict, List
import regex
import re
from tqdm import tqdm
import sqlite3
import importlib.util

from multiprocessing.util import Finalize
from multiprocessing import Pool as ProcessPool
from multiprocessing import cpu_count
from drqa.retriever import TfidfDocRanker
from drqa import tokenizers
from drqa.tokenizers import SimpleTokenizer
from drqa.retriever import DocDB
from drqa.retriever import utils

from nltk.tokenize import sent_tokenize

DEFAULT_CONVERT_CONFIG = {
    'tokenizer': SimpleTokenizer,
    'ranker': TfidfDocRanker,
    'db': DocDB,
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

PROCESS_TOK = None
PROCESS_DB = None
PROCESS_CANDS = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def _split_doc(doc: str):
    """
    Given a doc, split it into paragraphs.
    """
    curr = []
    curr_len = 0
    for split in regex.split(r'\n+', doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > 0:
            yield ' '.join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        yield ' '.join(curr)


class ConvertData2ParagraphClsInput(object):
    def __init__(self, tokenizer='', ranker_config=None, db_config=None, n_doc=5,
                 num_workers=1, convert_bs=48, ngram=2, small=False):

        self.convert_bs = convert_bs
        self.small = small
        self.n_doc = n_doc
        self.tok_class = tokenizers.get_class(tokenizer) if tokenizer else DEFAULT_CONVERT_CONFIG['tokenizer']
        self.annotators = set()
        self.tok_opts = {'annotators': self.annotators}
        self.ngram = ngram
        self.tokenizer = self.tok_class(**self.tok_opts)

        self.ranker_config = ranker_config if ranker_config else {}
        self.ranker_class = self.ranker_config.get('class', DEFAULT_CONVERT_CONFIG['ranker'])
        self.ranker_opt = self.ranker_config.get('ret_opt', {})
        logger.info('Loading ranker {}'.format(self.ranker_class.__name__))
        self.ranker = self.ranker_class(**self.ranker_opt)

        if hasattr(self.ranker, 'es'):
            self.db_config = ranker_config
            self.db_class =self.ranker_class
            self.db_opts = self.ranker_opts
        else:
            self.db_config = db_config or {}
            self.db_class = self.db_config.get('class', DEFAULT_CONVERT_CONFIG['db'])
            self.db_opts = self.db_config.get('db_opt', {})

        logger.info('Initializing tokenizers and document retrievers...')
        self.num_workers = num_workers
        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(self.tok_class, self.tok_opts, self.db_class, self.db_opts)
        )


    def read_input(self, data_file: str) -> List[Dict]:
        """
        Read the queries from data_file. The file should contain a list of Dicts,
        which at least should have two fields: entity and paragraph.
        """
        assert os.path.isfile(data_file)
        data_examples = json.load(open(data_file, 'r', encoding='utf8'))
        if self.small:
            data_examples = data_examples[:5]
        return data_examples


    def process(self, query, n_docs=5):
        """
        Run a single query.
        """
        para_examples = self.process_batch(
            [query], n_docs=n_docs
        )
        return para_examples[0]


    def process_batch(self, examples, n_docs=5):
        """
        Run a batch of queries' paragraphs (more efficient).
        """

        t0 = time.time()
        paragraphs = [example['paragraph'] for example in examples]
        prompts = [example['prompt'] for example in examples]
        queries = [prt + ' ' + para for para, prt in zip(paragraphs, prompts)]
        entities = [example['entity'] for example in examples]
        para_ids = [example['id'] for example in examples]
        topics = [example['topic'] for example in examples]

        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)
        # Rank documents for queries.
        if len(queries) == 1:
            ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.ranker.batch_closest_docs(
                queries, k=n_docs, num_workers=self.num_workers
            )
        all_docids, all_doc_scores = zip(*ranked)
        # all_doc_scores = [list(doc_scores) for doc_scores in all_doc_scores]

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(fetch_text, flat_docids)

        para_examples = []
        distant_positive_num = 0
        total_retrieval = 0

        for qidx in range(len(queries)):
            example = {'para_id': para_ids[qidx],
                       'entity': entities[qidx],
                       'topic': topics[qidx],
                       'prompt': prompts[qidx],
                       'paragraph': paragraphs[qidx]
                       }
            matched_para = []
            distant_num = 0

            for rel_didx, did in enumerate(all_docids[qidx]):
                didx = did2didx[did]

                if (len(queries[qidx]) > 0 and
                    len(doc_texts[didx].split()) > 2):

                    matched_para.append({
                        'wiki_id': did,
                        'para_id': rel_didx,
                        'tfidf_score': all_doc_scores[qidx][rel_didx],
                        'text': doc_texts[didx]
                    })

                    distant_num += 1

            if distant_num == 0:
                logger.info('Warning, question {} does not have distant answers in top {}.'.format(
                    queries[qidx], self.n_doc))
            else:
                total_retrieval += distant_num
                distant_positive_num += 1

            example['wiki'] = matched_para
            para_examples.append(example)

        logger.info('Reading %d questions...' % len(para_examples))

        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))
        logger.info('Average number of retrieval: {}'.format(total_retrieval / len(para_examples)))

        return para_examples, distant_positive_num


    def convert(self, data_file: str, output: str):
        """
        Convert input instances to retrieved wiki paragraphs
        """
        para_examples = []
        data_num = 0
        data_examples  = self.read_input(data_file=data_file)
        data_examples_batchs = [data_examples[data_id * self.convert_bs: (data_id + 1) * self.convert_bs]
                         for data_id in range(len(data_examples) // self.convert_bs + 1)]
        positive_total = 0

        for examples_per_batch in tqdm(data_examples_batchs, desc='convert batch examples',
                                       total=len(data_examples_batchs)):
            data_num += len(examples_per_batch)
            para_example_batch, positive_num = self.process_batch(examples_per_batch, n_docs=self.n_doc)
            positive_total += positive_num
            para_examples.extend(para_example_batch)

        assert len(data_examples) == data_num

        logger.info('Process {} examples to {} distant examples'.format(len(data_examples),
                                                                        len(para_examples),
                                                                        ))
        logger.info('Recall in top {} is {}'.format(self.n_doc, positive_total * 1.0 / data_num))
        json.dump(para_examples, open(output, 'w', encoding='utf8'), indent=4, ensure_ascii=False)
        print('Saved in JSON file.')


PROCESS_WIKI2PARA_DB = None
def init_wiki2para(db_class, db_opts):
    global PROCESS_WIKI2PARA_DB
    PROCESS_WIKI2PARA_DB = db_class(**db_opts)
    Finalize(PROCESS_WIKI2PARA_DB, PROCESS_WIKI2PARA_DB.close, exitpriority=100)

def fetch_text_wiki2para(doc_id):
    global PROCESS_WIKI2PARA_DB
    return PROCESS_WIKI2PARA_DB.get_doc_text(doc_id)

def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class ConvertWikiPage2Paragraph(object):
    def __init__(self, db_path=None, preprocess=None, num_workers=1):
        yes_no = input('Please confirm that you are going to convert a wikipage {} db to para db (yes/no):'.format(db_path))
        assert yes_no == 'yes'
        self.db = DocDB(db_path=db_path)
        self.all_doc_ids = self.db.get_doc_ids()
        self.num_workers = num_workers
        self.preprocess = preprocess
        self.processes = ProcessPool(
            num_workers,
            initializer=init_wiki2para,
            initargs=(DocDB, {})
        )
    def convert_page_to_paragraph(self, save_path):
        if os.path.isfile(save_path):
            raise RuntimeError('%s already exists! Not overwriting.' % save_path)

        logger.info('Start to fetch docs texts from doc db.')
        all_docs = list(tqdm(self.processes.imap(fetch_text_wiki2para, self.all_doc_ids),
                             desc='fetch docs', total=len(self.all_doc_ids)))
        assert len(all_docs) == len(self.all_doc_ids)
        doc_num = 0
        documents = []
        for doc_idx, doc in tqdm(enumerate(all_docs), desc='convert docs to paragraphs', total=len(all_docs)):
            para_num = 0
            doc_id = self.all_doc_ids[doc_idx]
            for para_split in _split_doc(doc):
                documents.append(('{}-{}'.format(doc_id, para_num), '{} {}'.format(doc_id, para_split)))
                para_num += 1
            doc_num += 1
        logger.info('Processed {} docs to paras.'.format(doc_num))
        logger.info('Start to insert to para db.')
        conn = sqlite3.connect(save_path)
        c = conn.cursor()
        c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

                # try:
        try:
            c.executemany("INSERT INTO documents VALUES (?,?)", documents)
        except:
            logger.info("Insert documents failed!")
                # except:
                #     logger.info('doc id {} cannot be inserted into database.'.format(pairs[0]))

        logger.info('Read %d paras.' % len(documents))
        logger.info('Committing...')
        conn.commit()
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_file', required=True, help='structura')
    parser.add_argument('-ret_path', required=True, help='retriver model path')
    parser.add_argument('-db_path', required=True, help='data base path')
    parser.add_argument('-output', required=True, help='output path')
    parser.add_argument('-batch', type=int, default=128, help='batch size in retrieval')
    parser.add_argument('-num_workers', default=None, help='number of processes')
    parser.add_argument('-n_doc', default=50, type=int, help='number of desired retrieval')
    parser.add_argument('-small', default=False, action='store_true', help='if true, use a tiny part of data for debugging')
    # parser.add_argument('--db_para_save_path', help='save path of paragraphs for db')
    opt = parser.parse_args()

    if 'para' in opt.ret_path:
        assert 'para' in opt.db_path
    convert2para = ConvertData2ParagraphClsInput(
                                                 num_workers=int(opt.num_workers) if opt.num_workers else None,
                                                 convert_bs=opt.batch,
                                                 ranker_config={'ret_opt': {'tfidf_path': opt.ret_path}},
                                                 db_config={'db_opt': {'db_path': opt.db_path}},
                                                 n_doc=opt.n_doc,
                                                 small=opt.small,
                                                 )
    convert2para.convert(data_file=opt.data_file, output=opt.output)

    # cwpp = ConvertWikiPage2Paragraph(db_path='/home/t-pinie/facebook-drqa/data/wikipedia/docs.db', num_workers=cpu_count())
    # cwpp.convert_page_to_paragraph(save_path=args.db_para_save_path)




