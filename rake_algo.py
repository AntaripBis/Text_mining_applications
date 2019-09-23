import spacy, json
from collections import Counter
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from statistics import mean

class RakeAlgo:

    GRAPH_EDGES = {}
    GRAPH_NODES = Counter()
    OPTIONS = {
                "thread_num": 10,
                "model_dictionary_type": "en_core_web_sm",
                "batch_len": 100
                }

    def _is_token_legit(self,token):
        return not token.is_stop and not token.like_num

    def _add_to_edges(self,token_set):
        temp_set = token_set.copy()
        for token in token_set:
            token_edges = self.GRAPH_EDGES[token] if token in self.GRAPH_EDGES else Counter()
            add_set = [temp_token for temp_token in temp_set if token != temp_token]
            token_edges.update(add_set)
            self.GRAPH_EDGES[token] = token_edges

    def _add_to_nodes(self,token_set):
        degree = 0.1 if len(token_set) == 1 else len(token_set) - 1
        add_counter = Counter({token:degree for token in token_set})
        self.GRAPH_NODES.update(add_counter)



    def _get_noun_phrase_cooccur(self,text,nlp_eng):
        phrase_list = [chunk for chunk in nlp_eng(text).noun_chunks]
        phrases = map(lambda x: set([token.text for token in x if self._is_token_legit(token)]),phrase_list)
        phrases = reduce(lambda a,x: a.union(x),phrases,set())
        if len(phrases) == 0:
            return
        self._add_to_nodes(phrases)
        return 1

    def _add_to_graph(self,text_list,nlp_eng=None):
        nlp_eng = spacy.load(self.OPTIONS["model_dictionary_type"]) if nlp_eng is None else nlp_eng
        with ThreadPoolExecutor(max_workers=self.OPTIONS["thread_num"]) as executor:
            for text in text_list:
                temp = executor.submit(self._get_noun_phrase_cooccur,text,nlp_eng)

    def _divide_corpus_to_subcorpus(self,corpus):
        batch_len = self.OPTIONS["batch_len"]
        start_idx = 0
        batch_count = int(ceil(len(corpus)/batch_len))
        for idx in range(batch_count):
            end_idx = (idx+1)*batch_len if len(corpus) > (idx+1)*batch_len else len(corpus)
            yield corpus[idx*batch_len:]

    def make_graph_from_corpus(self,corpus):
        nlp_eng = spacy.load(self.OPTIONS["model_dictionary_type"])
        for batch_corpus in self._divide_corpus_to_subcorpus(corpus):
            self._add_to_graph(batch_corpus,nlp_eng)

    def unload_graph(self,dump_file="tmp/token_graph.json"):
        with open(dump_file,"w") as f:
            json.dump(self.GRAPH_EDGES,f)

    def load_graph(self,dump_file="tmp/token_graph.json"):
        with open(dump_file,"r") as f:
            self.GRAPH_EDGES = json.load(f)

    def _update_counter(self,a,x):
        a.update(x)
        return a

    def compute_score_for_phrases(self,phrase_list):
        phrase_score_dict = []
        tokens = map(lambda x: Counter(x.split()),phrase_list)
        tokens = reduce(lambda a,x: self._update_counter(a,x),tokens,Counter())
        token_score = {token:(self.GRAPH_NODES[token] if token in self.GRAPH_NODES else 0.001)/freq for token,freq in tokens.items()}
        for phrase in phrase_list:
            phrase_score = mean([token_score[token] for token in phrase.split()])
            phrase_score_dict[phrase] = phrase_score

        return phrase_score_dict








