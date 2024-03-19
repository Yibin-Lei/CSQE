import argparse
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from pyserini.analysis import JDefaultEnglishAnalyzer, JWhiteSpaceAnalyzer
from pyserini.output_writer import OutputFormat, get_output_writer
from pyserini.pyclass import autoclass
from pyserini.query_iterator import get_query_iterator, TopicsFormat
from pyserini.search import JDisjunctionMaxQueryGenerator
from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher
from pyserini.search.lucene.reranker import ClassifierType, PseudoRelevanceClassifierReranker
from pyserini.search import get_qrels

import openai, json
from utils.openai_completion import OpenaiCompletion
from utils.common import save_json, save_jsonl, load_json, file_exists

from utils.eval_utils import TrecEvaluator

import spacy, random, re

# def load_spacy_nlp():
spacy_nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
from utils.trec_utils import load_trec

def truncate_text(text, max_len=128):
    return spacy_nlp(text)[:max_len].text[:(max_len * 10)]

def extract_key_sentences(response):
    pattern = r'"([^"]*)"'
    sentences = re.findall(pattern,response)
    joint_sentence = " ".join(sentences)
    return joint_sentence

class LuceneSearchInterface(object):
    def __init__(self, args):

        # 1. get searcher
        if not args.impact:
            if os.path.exists(args.index):
                # create searcher from index directory
                searcher = LuceneSearcher(args.index)
            else:
                # create searcher from prebuilt index name
                searcher = LuceneSearcher.from_prebuilt_index(args.index)
        elif args.impact:
            if os.path.exists(args.index):
                searcher = LuceneImpactSearcher(args.index, args.encoder, args.min_idf)
            else:
                searcher = LuceneImpactSearcher.from_prebuilt_index(args.index, args.encoder, args.min_idf)
        else:
            raise AttributeError("No searcher specified!")

        if args.language != 'en':
            searcher.set_language(args.language)

        if not searcher:
            exit()

        search_rankers = []

        if args.qld:
            search_rankers.append('qld')
            searcher.set_qld()
        elif args.bm25:
            search_rankers.append('bm25')
            if not args.disable_bm25_param:
                self.set_bm25_parameters(searcher, args.index, args.k1, args.b)

        if args.rm3:
            search_rankers.append('rm3')
            searcher.set_rm3()

        if args.rocchio:
            search_rankers.append('rocchio')
            if args.rocchio_use_negative:
                searcher.set_rocchio(gamma=0.15, use_negative=True)
            else:
                searcher.set_rocchio()

        # 2. text processor
        fields = dict()
        if args.fields:
            fields = dict([pair.split('=') for pair in args.fields])
            print(f'Searching over fields: {fields}')

        query_generator = None
        if args.dismax:
            query_generator = JDisjunctionMaxQueryGenerator(args.tiebreaker)
            print(f'Using dismax query generator with tiebreaker={args.tiebreaker}')

        if args.pretokenized:
            analyzer = JWhiteSpaceAnalyzer()
            searcher.set_analyzer(analyzer)
            if args.tokenizer is not None:
                raise ValueError(f"--tokenizer is not supported with when setting --pretokenized.")

        tokenizer = None
        if args.tokenizer != None:
            analyzer = JWhiteSpaceAnalyzer()
            searcher.set_analyzer(analyzer)
            print(f'Using whitespace analyzer because of pretokenized topics')
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            print(f'Using {args.tokenizer} to preprocess topics')

        if args.stopwords:
            analyzer = JDefaultEnglishAnalyzer.fromArguments('porter', False, args.stopwords)
            searcher.set_analyzer(analyzer)
            print(f'Using custom stopwords={args.stopwords}')

        # 3. get reranker
        ranker = None
        use_prcl = args.prcl and len(args.prcl) > 0 and args.alpha > 0
        if use_prcl is True:
            ranker = PseudoRelevanceClassifierReranker(
                searcher.index_dir, args.vectorizer, args.prcl, r=args.r, n=args.n, alpha=args.alpha)

        # Last. save to self
        self.args = args
        self.searcher, self.search_rankers = searcher, search_rankers
        self.fields, self.query_generator, self.tokenizer = fields, query_generator, tokenizer
        self.use_prcl, self.ranker = use_prcl, ranker

    @staticmethod
    def set_bm25_parameters(searcher, index, k1=None, b=None):
        if k1 is not None or b is not None:
            if k1 is None or b is None:
                print('Must set *both* k1 and b for BM25!')
                exit()
            print(f'Setting BM25 parameters: k1={k1}, b={b}')
            searcher.set_bm25(k1, b)
        else:
            # Automatically set bm25 parameters based on known index...
            if index == 'msmarco-passage' or index == 'msmarco-passage-slim' or index == 'msmarco-v1-passage' or \
                    index == 'msmarco-v1-passage-slim' or index == 'msmarco-v1-passage-full':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-passage.md
                print('MS MARCO passage: setting k1=0.82, b=0.68')
                searcher.set_bm25(0.82, 0.68)
            elif index == 'msmarco-passage-expanded' or \
                    index == 'msmarco-v1-passage-d2q-t5' or \
                    index == 'msmarco-v1-passage-d2q-t5-docvectors':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-passage-docTTTTTquery.md
                print('MS MARCO passage w/ doc2query-T5 expansion: setting k1=2.18, b=0.86')
                searcher.set_bm25(2.18, 0.86)
            elif index == 'msmarco-doc' or index == 'msmarco-doc-slim' or index == 'msmarco-v1-doc' or \
                    index == 'msmarco-v1-doc-slim' or index == 'msmarco-v1-doc-full':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc.md
                print('MS MARCO doc: setting k1=4.46, b=0.82')
                searcher.set_bm25(4.46, 0.82)
            elif index == 'msmarco-doc-per-passage' or index == 'msmarco-doc-per-passage-slim' or \
                    index == 'msmarco-v1-doc-segmented' or index == 'msmarco-v1-doc-segmented-slim' or \
                    index == 'msmarco-v1-doc-segmented-full':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-segmented.md
                print('MS MARCO doc, per passage: setting k1=2.16, b=0.61')
                searcher.set_bm25(2.16, 0.61)
            elif index == 'msmarco-doc-expanded-per-doc' or \
                    index == 'msmarco-v1-doc-d2q-t5' or \
                    index == 'msmarco-v1-doc-d2q-t5-docvectors':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-docTTTTTquery.md
                print('MS MARCO doc w/ doc2query-T5 (per doc) expansion: setting k1=4.68, b=0.87')
                searcher.set_bm25(4.68, 0.87)
            elif index == 'msmarco-doc-expanded-per-passage' or \
                    index == 'msmarco-v1-doc-segmented-d2q-t5' or \
                    index == 'msmarco-v1-doc-segmented-d2q-t5-docvectors':
                # See https://github.com/castorini/anserini/blob/master/docs/regressions-msmarco-doc-segmented-docTTTTTquery.md
                print('MS MARCO doc w/ doc2query-T5 (per passage) expansion: setting k1=2.56, b=0.59')
                searcher.set_bm25(2.56, 0.59)

    def get_setting_name(self):
        tokens = ['run', '+'.join(self.search_rankers)]
        setting_name = '.'.join(tokens)

        if self.use_prcl is True:
            clf_rankers = []
            for t in self.args.prcl:
                if t == ClassifierType.LR:
                    clf_rankers.append('lr')
                elif t == ClassifierType.SVM:
                    clf_rankers.append('svm')
            r_str = f'prcl.r_{self.args.r}'
            n_str = f'prcl.n_{self.args.n}'
            a_str = f'prcl.alpha_{self.args.alpha}'
            clf_str = 'prcl_' + '+'.join(clf_rankers)
            tokens2 = [self.args.vectorizer, clf_str, r_str, n_str, a_str]
            setting_name += ('-' + '-'.join(tokens2))

        return setting_name

    def do_retrieval(self, text, num_hits, num_thread=12):
        if isinstance(text, str):
            if (self.args.tokenizer != None):
                toks = self.tokenizer.tokenize(text)
                text = ' '
                text = text.join(toks)
            if self.args.impact:
                hits = self.searcher.search(text, num_hits, fields=self.fields)
            else:
                hits = self.searcher.search(text, num_hits, query_generator=self.query_generator, fields=self.fields)
            results = hits
        else:
            batch_topics = text  # batch of queries
            pseudo_batch_topic_ids = [str(idx) for idx, _ in enumerate(text)]

            if (self.args.tokenizer != None):
                new_batch_topics = []
                for text in batch_topics:
                    toks = self.tokenizer.tokenize(text)
                    text = ' '
                    text = text.join(toks)
                    new_batch_topics.append(text)
                batch_topics = new_batch_topics

            if self.args.impact:
                results = self.searcher.batch_search(
                    batch_topics, pseudo_batch_topic_ids, num_hits, num_thread, fields=self.fields,
                )
            else:
                results = self.searcher.batch_search(
                    batch_topics, pseudo_batch_topic_ids, num_hits, num_thread,
                    query_generator=self.query_generator, fields=self.fields
                )
            results = [results[id_] for id_ in pseudo_batch_topic_ids]
        return results

    def do_rerank(self, hits):
        # do rerank
        if self.use_prcl and len(hits) > (self.args.r + self.args.n):
            docids = [hit.docid.strip() for hit in hits]
            scores = [hit.score for hit in hits]
            scores, docids = self.ranker.rerank(docids, scores)
            docid_score_map = dict(zip(docids, scores))
            for hit in hits:
                hit.score = docid_score_map[hit.docid.strip()]
        return hits

    def do_postprocess(self, hits, topic_id, remove_duplicates=False, remove_query=False):
        if remove_duplicates:
            seen_docids = set()
            dedup_hits = []
            for hit in hits:
                if hit.docid.strip() in seen_docids:
                    continue
                seen_docids.add(hit.docid.strip())
                dedup_hits.append(hit)
            hits = dedup_hits

        # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
        # We want to remove the query from the results.
        if remove_query:
            hits = [hit for hit in hits if hit.docid != topic_id]
            print("remove query")

        return hits


def search_iterator(args, search_api, qid_query_list):
    batch_topics = list()
    batch_topic_ids = list()
    for index, (topic_id, text) in enumerate(tqdm(qid_query_list, total=len(qid_query_list))):
        if args.batch_size <= 1 and args.threads <= 1:
            hits = search_api.do_retrieval(text, args.hits)
            results = [(topic_id, hits)]
        else:
            batch_topic_ids.append(topic_id)
            batch_topics.append(text)
            if (index + 1) % args.batch_size == 0 or index == len(qid_query_list) - 1:
                hits_list = search_api.do_retrieval(batch_topics, args.hits, args.threads)
                results = [(id_, hits) for id_, hits in zip(batch_topic_ids, hits_list)]

                batch_topic_ids.clear(), batch_topics.clear()
            else:
                continue
        # post process
        for topic_id, hits in results:
            hits = search_api.do_rerank(hits)
            hits = search_api.do_postprocess(
                hits, topic_id, args.remove_duplicates, args.remove_query)
            yield topic_id, hits
        results.clear()


def define_search_args(parser):
    parser.add_argument('--index', type=str, metavar='path to index or index name', required=True,
                        help="Path to Lucene index or name of prebuilt index.")

    parser.add_argument('--impact', action='store_true', help="Use Impact.")
    parser.add_argument('--encoder', type=str, default=None, help="encoder name")
    parser.add_argument('--min-idf', type=int, default=0, help="minimum idf")

    parser.add_argument('--bm25', action='store_true', default=True, help="Use BM25 (default).")
    parser.add_argument('--k1', type=float, help='BM25 k1 parameter.')
    parser.add_argument('--b', type=float, help='BM25 b parameter.')

    parser.add_argument('--rm3', action='store_true', help="Use RM3")
    parser.add_argument('--rocchio', action='store_true', help="Use Rocchio")
    parser.add_argument('--rocchio-use-negative', action='store_true', help="Use nonrelevant labels in Rocchio")
    parser.add_argument('--qld', action='store_true', help="Use QLD")

    parser.add_argument('--language', type=str, help='language code for BM25, e.g. zh for Chinese', default='en')
    parser.add_argument('--pretokenized', action='store_true', help="Boolean switch to accept pre-tokenized topics")

    parser.add_argument('--prcl', type=ClassifierType, nargs='+', default=[],
                        help='Specify the classifier PseudoRelevanceClassifierReranker uses.')
    parser.add_argument('--prcl.vectorizer', dest='vectorizer', type=str,
                        help='Type of vectorizer. Available: TfidfVectorizer, BM25Vectorizer.')
    parser.add_argument('--prcl.r', dest='r', type=int, default=10,
                        help='Number of positive labels in pseudo relevance feedback.')
    parser.add_argument('--prcl.n', dest='n', type=int, default=100,
                        help='Number of negative labels in pseudo relevance feedback.')
    parser.add_argument('--prcl.alpha', dest='alpha', type=float, default=0.5,
                        help='Alpha value for interpolation in pseudo relevance feedback.')

    parser.add_argument('--fields', metavar="key=value", nargs='+',
                        help='Fields to search with assigned float weights.')
    parser.add_argument('--dismax', action='store_true', default=False,
                        help='Use disjunction max queries when searching multiple fields.')
    parser.add_argument('--dismax.tiebreaker', dest='tiebreaker', type=float, default=0.0,
                        help='The tiebreaker weight to use in disjunction max queries.')

    parser.add_argument('--stopwords', type=str, help='Path to file with customstopwords.')


def progressive_query_rewrite(
        openai_api, query, top_passages,
        max_demo_len=None, index=None,
        expansion_method="",
        *arg, **kwargs):

    if max_demo_len:
        top_passages = [truncate_text(psg, max_demo_len) for psg in top_passages]

    top_passages_str = "\n".join([f"{idx+1}. {psg}" for idx, psg in enumerate(top_passages)])

    sys_prompts = {

        "keqe": "You are a helpful assistant.",

        "csqe": "You are an efficient search assistant. I will provide you with a query and its initially retrieved documents.\
        You will begin by examining the initially retrieved documents and identifying the ones that are relevant, even partially, to the query.\
        Once the relevant documents are identified, you will extract the key sentences from each document that contribute to their relevance.",

    }

    example_user_prompt = """Query: \"how are some sharks warm blooded\"
Retrieved documents:
1. Most sharks are cold-blooded. Some, like the Mako and the Great white shark, are partially warmblooded (they are endotherms). Cold blooded although if you've ever seen a Great White Shark hunt sea lions you'd be thinking they would have to be hotblooded. Actually the Salmon Shark is a warm blooded shark.
2. Are sharks cold-blooded or warm-blooded? Sharks have a reputation as cold-blooded and despite how negative that term is, it is not entirely inaccurate. Sharks are by no means evil, vicious killers like that quote suggests. Nonetheless, sharks are, for the most part anyways, efficient ectothermic predators. Endo vs Ecto.
3. Great white sharks are some of the only warm blooded sharks. This allows them to swim in colder waters in addition to warm, tropical waters. Great White sharks can be found asâ¦ north as Alaska and as south as the southern tip of South America. They exist worldwide, everywhere in-between. 5 people found this useful.
4. Sharks' blood gives them turbo speed. Several species of shark and tuna have something special going on inside their bodies. For a long time, scientists have known that some fish species appear warm-blooded. Salmon sharks can elevate their body temperatures by up to 20 degrees compared to the surrounding water, for example.

You will begin by examining the initially retrieved documents and identifying the ones that are relevant, even partially, to the query. Once the relevant documents are identified, you will extract the key sentences from each document that contribute to their relevance."""

    example_sys_prompt = """Based on the query \"how are some sharks warm blooded\", I have examined the initially retrieved documents. Here are the relevant documents and the key sentences extracted from each: 
Document 1:
\"Most sharks are cold-blooded. Some, like the Mako and the Great white shark, are partially warm-blooded (they are endotherms).\"
\"Actually, the Salmon Shark is a warm-blooded shark.\"

Document 3:
\"Great white sharks are some of the only warm-blooded sharks.\"
\"This allows them to swim in colder waters in addition to warm, tropical waters.\"

Document 4:
\"Salmon sharks can elevate their body temperatures by up to 20 degrees compared to the surrounding water, for example.\""""

    user_prompts = {
        "keqe" : f"""Please write a passage to answer the question
Question: {query}
Passage:""",


        "csqe": f"""Query: \"{query}\"
Retrieved documents:
{top_passages_str}

You will begin by examining the initially retrieved documents and identifying the ones that are relevant, even partially, to the query. Once the relevant documents are identified, you will extract the key sentences from each document that contribute to their relevance.""",

    }
    
    if expansion_method == "csqe":
        # keqe expansion
        messages = [
        {"role": "system", "content": sys_prompts["keqe"]},
        {"role": "user", "content": user_prompts["keqe"]},

        ]
        gen_fn = openai_api.completion_chat
        response_list_keqe = gen_fn(messages, *arg, **kwargs)

        # corpus-originated expansion
        messages = [
            {"role": "system", "content": sys_prompts["csqe"]},
            {"role": "user", "content": example_user_prompt},
            {"role": "assistant", "content": example_sys_prompt},
            {"role": "user", "content": user_prompts["csqe"]},

        ]
        gen_fn = openai_api.completion_chat
        response_list_csqe = gen_fn(messages, *arg, **kwargs)
        response_list = response_list_keqe + response_list_csqe
        print(response_list)

        new_list = [query] * len(response_list_keqe) + response_list_keqe + [extract_key_sentences(response_csqe) for response_csqe in response_list_csqe]

    elif expansion_method == "keqe":
        messages = [
        {"role": "system", "content": sys_prompts["keqe"]},
        {"role": "user", "content": user_prompts["keqe"]},

        ]
        gen_fn = openai_api.completion_chat
        response_list = gen_fn(messages, *arg, **kwargs)
        print(response_list)

        new_list = [query] * len(response_list) + response_list

    user_query = "\n".join(new_list).lower()

    return user_query, response_list


def main():
    JLuceneSearcher = autoclass('io.anserini.search.SimpleSearcher')
    parser = argparse.ArgumentParser(description='Search a Lucene index.')
    define_search_args(parser)
    parser.add_argument('--topics', type=str, metavar='topic_name', required=True,
                        help="Name of topics. Available: robust04, robust05, core17, core18.")
    parser.add_argument('--hits', type=int, metavar='num',
                        required=False, default=1000, help="Number of hits.")
    parser.add_argument('--topics-format', type=str, metavar='format', default=TopicsFormat.DEFAULT.value,
                        help=f"Format of topics. Available: {[x.value for x in list(TopicsFormat)]}")
    parser.add_argument('--output-format', type=str, metavar='format', default=OutputFormat.TREC.value,
                        help=f"Format of output. Available: {[x.value for x in list(OutputFormat)]}")
    parser.add_argument('--output', type=str, metavar='path',
                        help="Path to output file.")
    parser.add_argument('--max-passage', action='store_true',
                        default=False, help="Select only max passage from document.")
    parser.add_argument('--max-passage-hits', type=int, metavar='num', required=False, default=100,
                        help="Final number of hits when selecting only max passage.")
    parser.add_argument('--max-passage-delimiter', type=str, metavar='str', required=False, default='#',
                        help="Delimiter between docid and passage id.")
    parser.add_argument('--batch-size', type=int, metavar='num', required=False,
                        default=1, help="Specify batch size to search the collection concurrently.")
    parser.add_argument('--threads', type=int, metavar='num', required=False,
                        default=1, help="Maximum number of threads to use.")
    parser.add_argument('--tokenizer', type=str, help='tokenizer used to preprocess topics')
    parser.add_argument('--remove-duplicates', action='store_true', default=False, help="Remove duplicate docs.")
    # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
    # We want to remove the query from the results. This is equivalent to -removeQuery in Java.
    parser.add_argument('--remove_query', action='store_true', default=False, help="Remove query from results list.")

    # new
    parser.add_argument('--disable_bm25_param', action='store_true', default=True, help="Use BM25 (default).")
    parser.add_argument('--qrels', type=str, metavar='qrels_name', default=None,
                        help="In case of the difference between topics and qrels")
    parser.add_argument('--metric_beir', action='store_true', help="default false to ms-marco trec")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save outputs")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite existing output dir")
    parser.add_argument("--openai_api_key", type=str, default="none")
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo-0301")
    parser.add_argument("--answer_key", type=str, default="contents")
    parser.add_argument("--keep_passage_num", type=int, default=10, help="Number of passages kept for CSQE")
    parser.add_argument('--write_top_passages', action='store_true', help="Save the top retrieved passages")
    parser.add_argument("--gen_num", type=int, default=5, help="Number of query expansions to generate")
    parser.add_argument('--max_demo_len', type=int, default=None, help="Truncation length for each retrieved passage")
    parser.add_argument('--max_tokens', type=int, default=1024, help="Maximum number of tokens to generate each time")
    parser.add_argument('--expansion_method', type=str, default="csqe")
    parser.add_argument('--trec_python_path', type=str, default="python3")



    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # Create output directory if needed

    # openai stuff
    openai.api_key = args.openai_api_key
    openai_api = OpenaiCompletion(model_name=args.openai_model)

    # query data
    query_iterator = get_query_iterator(args.topics, TopicsFormat(args.topics_format))
    qrels_name = args.qrels or args.topics
    qrels = get_qrels(qrels_name)
    org_qid_query_list = [(k, v) for k, v in query_iterator if k in qrels]  # topics = query_iterator.topics
    print(f"The number of query is {len(org_qid_query_list)}")

    # my own api
    search_api = LuceneSearchInterface(args)
    trec_eval = TrecEvaluator(args.trec_python_path)

    # round 1 - regular bm25
    last_top_passages = dict()  # qid2passages
    iter_org_qid_query_list = org_qid_query_list

    flag_gen_aug = False
    for ridx in range(2):
        output_path_bm25 = os.path.join(args.output_dir, f'bm25-aug{ridx}_result_retrieval.trec')
        if flag_gen_aug:
            # # data generation
            aug_qid_query_list = []
            qid2responses = dict()

            for index, (topic_id, query) in enumerate(tqdm(org_qid_query_list, total=len(org_qid_query_list))):
                top_passages = last_top_passages[topic_id][
                               0:(0 + args.keep_passage_num)]

                query_aug, response_list = progressive_query_rewrite(
                    openai_api, query, top_passages, n=args.gen_num,
                    max_demo_len=args.max_demo_len, index=args.index,
                    expansion_method=args.expansion_method,
                    temperature=1.0, max_tokens=args.max_tokens,
                )
                aug_qid_query_list.append((topic_id, query_aug))
                print(topic_id, "|||", query_aug)

                qid2responses[topic_id] = response_list

            save_json(qid2responses, output_path_bm25 + ".responses.json")
            iter_org_qid_query_list = aug_qid_query_list
    
        # save query
        save_jsonl(iter_org_qid_query_list, output_path_bm25 + ".topics.jsonl")

        output_writer = get_output_writer(
            output_path_bm25, OutputFormat(args.output_format),
            'w', max_hits=args.hits, tag='Anserini', topics=query_iterator.topics, use_max_passage=args.max_passage,
            max_passage_delimiter=args.max_passage_delimiter, max_passage_hits=args.max_passage_hits)

        with output_writer:
            for topic_id, hits in search_iterator(args, search_api, iter_org_qid_query_list):
                output_writer.write(topic_id, hits)
                # save passage result
                passages = []
                for hit in hits:
                    raw_df = json.loads(hit.raw)
                    text_list = [raw_df[k] for k in args.answer_key.split("|") if raw_df[k]]
                    passages.append("\t".join(text_list))
                last_top_passages[topic_id] = passages

        # save top passages
        if args.write_top_passages:
            save_json(last_top_passages, output_path_bm25 + "top-psgs.json")

        if args.metric_beir:
            result_metrics = trec_eval.predefined_beir(qrels_name, output_path_bm25)
        else:
            result_metrics = trec_eval.predefined_msmarco_trec(qrels_name, output_path_bm25)
        save_json(result_metrics, output_path_bm25 + ".metrics.json")
        print(f"At round {ridx}: {json.dumps(result_metrics)}")

        flag_gen_aug = True


if __name__ == '__main__':
    main()








