import os


class TrecEvaluator(object):
    def __init__(self, python_interpreter="python3", verbose=False):
        self.base_cmd = f"{python_interpreter} -m pyserini.eval.trec_eval"
        self.verbose = verbose

    def extract_from_stdout(self, p):
        last_line = None
        for line in p.readlines():
            last_line = line
        metric = float(last_line.split("\t")[-1].strip())
        return metric

    def calc_trec_metric(self, cmd, qrels_name, trec_name, ):
        final_cmd = f"{self.base_cmd} -c {cmd} {qrels_name} {trec_name}"
        print(f"run {final_cmd}")
        p = os.popen(final_cmd)
        return self.extract_from_stdout(p)

    def predefined_msmarco_trec(self, qrels_name, trec_name):
        return {
            "map": self.calc_trec_metric("-l 2 -m map", qrels_name, trec_name, ),
            "ndcg@10": self.calc_trec_metric("-m ndcg_cut.10", qrels_name, trec_name, ),
            "recall@1000": self.calc_trec_metric("-l 2 -m recall.1000", qrels_name, trec_name, ),

            # for analyzing purpose
            "recall@1": self.calc_trec_metric("-l 2 -m recall.1", qrels_name, trec_name, ),
            "recall@5": self.calc_trec_metric("-l 2 -m recall.5", qrels_name, trec_name, ),
            "recall@10": self.calc_trec_metric("-l 2 -m recall.10", qrels_name, trec_name, ),
            "recall@20": self.calc_trec_metric("-l 2 -m recall.20", qrels_name, trec_name, ),
        }

    def predefined_beir(self, qrels_name, trec_name):
        return {
            "ndcg.10": self.calc_trec_metric("-m ndcg_cut.10", qrels_name, trec_name, ),
            "recall.100": self.calc_trec_metric("-m recall.100", qrels_name, trec_name, ),
            "recall.1000": self.calc_trec_metric("-m recall.1000", qrels_name, trec_name, ),

            # for analyzing purpose
            "recall@1": self.calc_trec_metric("-m recall.1", qrels_name, trec_name, ),
            "recall@5": self.calc_trec_metric("-m recall.5", qrels_name, trec_name, ),
            "recall@10": self.calc_trec_metric("-m recall.10", qrels_name, trec_name, ),
            "recall@20": self.calc_trec_metric("-m recall.20", qrels_name, trec_name, ),
        }
