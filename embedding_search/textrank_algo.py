import networkx as nx
import spacy, json
from collections import Counter
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from statistics import mean

class TextRank:

    def __init__(self):
        self.GRAPH_EDGES = {}
        self.GRAPH_NODES = Counter()
        self.OPTIONS = {
                        "thread_num": 20,
                        "model_dictionary_type": "en_core_web_sm",
                        "batch_len": 500,
                        "alpha": 0.9,
                        "max_iterations": 100
                        }


    def compute_text_rank(self,is_digraph=True):
        graph = nx.DiGraph(self.GRAPH_EDGES) if is_digraph else nx.Graph(self.GRAPH_EDGES)
        nx.info(graph)
        pagerank = nx.pagerank(self.GRAPH,alpha=self.OPTIONS["alpha"],
                                    max_iter=self.OPTIONS["max_iterations"])
        return pagerank

    def _is_token_legit(self,token):
        return not token.is_stop and not token.like_num

    def _add_to_edges(self,token_set):
        temp_set = token_set.copy()
        for token in token_set:
            token_edges = self.GRAPH_EDGES[token] if token in self.GRAPH_EDGES else Counter()
            add_set = [temp_token for temp_token in temp_set if token != temp_token]
            token_edges.update(add_set)
            self.GRAPH_EDGES[token] = token_edges

    def _get_noun_phrase_cooccur(self,text,nlp_eng):
        phrase_list = [chunk for chunk in nlp_eng(text).noun_chunks]
        phrases = map(lambda x: set([token.text for token in x if self._is_token_legit(token)]),phrase_list)
        phrases = reduce(lambda a,x: a.union(x),phrases,set())
        if len(phrases) == 0:
            return
        if len(phrases) == 1:
            phrases.add("NO-PHRASE")
        # self._add_to_nodes(phrases)
        self._add_to_edges(phrases)
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