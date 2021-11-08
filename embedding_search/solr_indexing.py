from multiprocessing import Pool,cpu_count
from math import ceil

import pysolr

def create_solr_instance(solr_url='http://localhost:8983/solr/',timeout=60):
    solr = pysolr.Solr(solr_url, always_commit=True, timeout=timeout)
    solr.ping()
    return solr

def load_batch_data(data,batch_num: int,batch_size: int):
    start_idx = batch_num*batch_size
    end_idx = (batch_num+1)*batch_size
    end_idx = end_idx if end_idx < len(data) else len(data)
    batch_data = [{"doc_text": text} for text in data[start_idx:end_idx]]
    return batch_data

def create_index_parallel(solr,data: list):
    n_partition = cpu_count()
    batch_size = int(ceil(len(data)/n_partition))
    data_batches = [load_batch_data(data,i,batch_size) for i in n_partition] 
    
    def add_docs_to_index(batch_data):
        solr.add(batch_data)
        return 1
        
    p = Pool(n_partition)
    
    result = p.imap(add_docs_to_index,data_batches)
    result = list(result)
    p.close()
    p.join()
    return result[:100] if 100 < len(result) else len(result)


def create_index_parallel(solr,data: list,n_partition=100):
    batch_size = int(ceil(len(data)/n_partition))
    data_batches = [load_batch_data(data,i,batch_size) for i in n_partition] 
    
    def add_docs_to_index(batch_data):
        solr.add(batch_data)
        return 1
        
    result = [add_docs_to_index(batch_data) for batch_data in data_batches]
        
    return result[:100] if 100 < len(result) else len(result)
    
        
    
        