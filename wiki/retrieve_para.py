import time
import os
import json
import argparse
import logging
import regex
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

def has_answer(answer, para_text, match, tokenizer):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    text = tokenizer.tokenize(para_text).words(uncased=True)
    if match == 'string':
        # Answer is a list of possible string
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    else:
        raise ValueError('wrong match type!')
    return False

def _split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
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
                 num_workers=1, convert_bs=48, ngram=2, distant=False, small=False):

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

        self.distant = distant

    def read_data_txt(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as fin:
            return [json.loads(line.strip('\n')) for line in fin.readlines()]

    def read_input(self, data_file):
        assert os.path.isfile(data_file) is not False
        data_examples = self.read_data_txt(data_file=data_file)
        data_examples = self.filter_query(data_examples)
        if self.small:
            data_examples = data_examples[:200]
        return data_examples

    def filter_query(self, data_examples):
        new_data_examples = []
        for example in tqdm(data_examples, desc='filter queries', total=len(data_examples)):
            query_tokens = self.tokenizer.tokenize(example['question'])
            query_words = query_tokens.ngrams(n=self.ngram, filter_fn=utils.filter_ngram)
            if len(query_words) != 0:
                new_data_examples.append(example)
        logger.info('Process {} queries to {}'.format(len(data_examples), len(new_data_examples)))
        return new_data_examples[:]

    def process(self, query, n_docs=5,
                return_context=False):
        """Run a single query."""
        para_examples = self.process_batch(
            [query], n_docs=n_docs
        )
        return para_examples[0]

    def process_batch(self, examples, n_docs=5, mode='train'):
        """Run a batch of queries' paragraphs (more efficient)."""
        t0 = time.time()
        queries = [example['question'] for example in examples]
        gold_paras = [example['squad_context'] for example in examples]
        answers = [example['answer'] for example in examples]
        squad_ids = [example['squad_qid'] for example in examples]
        squad_contexts = [example['squad_context'] for example in examples]
        squad_answers = [example['squad_answers'] for example in examples]

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

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        # didx2sidx: List[doc_idx -> [split_idx_start, split_idx_end]]
        flat_splits = []
        didx2sidx = []
        for text in doc_texts:
            splits = _split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)
        if len(flat_splits) != doc_texts:
            logger.info('{} doc texts are splited into {} splits'.format(len(doc_texts), len(flat_splits)))
        para_examples = []
        distant_positive_num = 0
        for qidx in range(len(queries)):
            para_example = []
            distant_num = 0
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    if (len(queries[qidx]) > 0 and
                        len(flat_splits[sidx].split()) > 2) and not has_answer(answer=answers[qidx],
                                                                               para_text=flat_splits[sidx],
                                                                               match='string',
                                                                               tokenizer=self.tokenizer):
                        para_example.append({

                            'id': (qidx, did, rel_didx, sidx),
                            'question': queries[qidx],
                            'document': flat_splits[sidx],
                            'label': '0',
                            'squad_id': squad_ids[qidx],
                            'squad_answers': squad_answers[qidx],
                            'squad_context': squad_contexts[qidx],
                            'tfidf_score': all_doc_scores[qidx][rel_didx],
                        })
                    elif (len(queries[qidx]) > 0 and
                        len(flat_splits[sidx].split()) > 2) and  has_answer(answer=answers[qidx],
                                                                               para_text=flat_splits[sidx],
                                                                               match='string',
                                                                               tokenizer=self.tokenizer) and self.distant:
                        para_example.append({

                            'id': (qidx, did, rel_didx, sidx),
                            'question': queries[qidx],
                            'document': flat_splits[sidx],
                            'label': '1',
                            'squad_id': squad_ids[qidx],
                            'squad_answers': squad_answers[qidx],
                            'squad_context': squad_contexts[qidx],
                            'tfidf_score': all_doc_scores[qidx][rel_didx],
                        })
                        distant_num += 1

            if not self.distant:
                para_example.append({

                        'id': (qidx, -1, -1),
                        'question': queries[qidx],
                        'document': gold_paras[qidx],
                        'label': '1',
                        'squad_id': squad_ids[qidx],
                        'squad_answers': squad_answers[qidx],
                        'squad_context': squad_contexts[qidx],
                        'tfidf_score': 1000000.0,
                    })
            if self.distant and distant_num == 0:
                logger.info('Warning, question {} does not have distant answers in top {}.'.format(
                    queries[qidx], self.n_doc))
                if mode == 'train':
                    logger.info('skiped this query.')
                    continue
            if distant_num != 0:
                distant_positive_num += 1
            para_examples.append(para_example)

        logger.info('Reading %d questions...' % len(para_examples))

        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))

        return para_examples, distant_positive_num


    def write_data_paras(self, para_examples, data_file):
        logger.info('write paras of data file to {}'.format(data_file + '.para'))
        with open(data_file + '.para', 'w', encoding='utf-8') as fout:
            for para_examples_batch in tqdm(para_examples, desc='write paras', total=len(para_examples)):
                for para_example in para_examples_batch:
                    fout.write(json.dumps(para_example, ensure_ascii=False))
                    fout.write('\n')

    def convert(self, data_file, mode='train'):
        para_examples = []
        data_num = 0
        data_examples  = self.read_input(data_file=data_file)
        data_examples_batchs = [data_examples[data_id * self.convert_bs: (data_id + 1) * self.convert_bs]
                         for data_id in range(len(data_examples) // self.convert_bs + 1)]
        positive_total = 0
        for examples_per_batch in tqdm(data_examples_batchs, desc='convert batch examples',
                                       total=len(data_examples_batchs)):
            data_num += len(examples_per_batch)
            para_example_batch, positive_num = self.process_batch(examples_per_batch, n_docs=self.n_doc, mode=mode)
            positive_total += positive_num
            para_examples.extend(para_example_batch)
        assert len(data_examples) == data_num

        if self.distant:
            logger.info('Process {} examples to {} distant examples, in mode: {}'.format(len(data_examples),
                                                                                         len(para_examples),
                                                                                         mode))
            logger.info('Recall in top {} is {}'.format(self.n_doc, positive_total * 1.0 / data_num))
        self.write_data_paras(para_examples, data_file=data_file)


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
    parser.add_argument('--data_file', help='data file to convert to paras for classification.', required=True)
    parser.add_argument('--ret_path', help='retriver model path')
    parser.add_argument('--db_path', help='data base path')
    parser.add_argument('--num_workers', default=None)
    parser.add_argument('--n_para', default=50, type=int, help='para number of retrieval')
    parser.add_argument('--distant', default=1, type=int)
    parser.add_argument('--small', default=0, type=int)
    # parser.add_argument('--db_para_save_path', help='save path of paragraphs for db')
    args = parser.parse_args()
    if 'dev' in args.data_file or 'test' in args.data_file:
        mode = 'dev'
    elif 'train' in args.data_file:
        mode = 'train'
    else:
        raise ValueError('data file did not contain dev or train')
    if 'para' in args.ret_path:
        assert 'para' in args.db_path
    convert2para = ConvertData2ParagraphClsInput(
                                                 num_workers=int(args.num_workers) if args.num_workers else None,
                                                 convert_bs=128,
                                                 ranker_config={'ret_opt': {'tfidf_path': args.ret_path}},
                                                 db_config={'db_opt': {'db_path': args.db_path}},
                                                 n_doc=args.n_para,
                                                 small=bool(args.small),
                                                 distant=bool(args.distant),
                                                 )
    convert2para.convert(args.data_file, mode)

    # cwpp = ConvertWikiPage2Paragraph(db_path='/home/t-pinie/facebook-drqa/data/wikipedia/docs.db', num_workers=cpu_count())
    # cwpp.convert_page_to_paragraph(save_path=args.db_para_save_path)




