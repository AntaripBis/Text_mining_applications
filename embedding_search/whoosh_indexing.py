from multiprocessing import Pool,cpu_count

import pandas as pd

from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer,StemmingAnalyzer

INDEX_MEMORY_LIMIT = 1024
def create_schema():
    schema = Schema(doc_text=TEXT(analyzer = StemmingAnalyzer(),stored=True))
    return schema


def create_index(data: pd.DataFrame,text_col:str,index_path: str):
    schema = create_schema()
    ix = index.create_in(index_path,schema)
    print("CPU count : %d " % (cpu_count()))
    writer = ix.writer(limitmb=INDEX_MEMORY_LIMIT,procs=cpu_count(),multisegment=True)
    
    def add_text(text):
        writer.add_document(doc_text=text)
        return 1
     
    
    return_list = [add_text(x) for x in list(data[text_col])]
    print(return_list[:100 if len(return_list) > 100 else len(return_list)])
    writer.commit()
    return ix

def search_index(ix,query_text: str,result_num: int=20):
    parser = QueryParser("doc_text", ix.schema)
    query = parser.parse(query_text)
    
    results = []
    with ix.searcher() as s:
        result_iterator = s.search(query,limit=result_num,terms=True)
        results = [[result.keys(),result.values(),result.rank,result.score] for result in result_iterator]
    
    print(len(results))
    return results