from math import ceil

import torch
from transformers import RobertaModel,RobertaTokenizer


__all__ = ["load_roberta","add_pad","make_attn_mask","encode_text"]

def load_roberta(model_type="roberta-base",is_eval: bool=True):
    model = RobertaModel.from_pretrained(model_type)
    if is_eval:
        model.eval()
    tokenizer = RobertaTokenizer.from_pretrained(model_type,eos_token="[SEP]",sep_token="[SEP]",
                                                 cls_token="[CLS]",pad_token="[PAD]")
    return model,tokenizer

def add_pad(tokens: list,seq_len: int=512):
    pads = [] if len(tokens) >= seq_len else ["[PAD]"]*(seq_len - len(tokens))
    return tokens+pads

def divide_tokens_into_batches(tokens: list,batch_size: int=512):
    if len(tokens) < batch_size:
        return [add_pad(tokens)]
    n_batches = int(ceil(len(tokens)/batch_size))
    tokens_batches = []
    for i in range(n_batches):
        if i != n_batches-1:
            end_idx = (i+1)*batch_size-(1 if i == 0 else 2)
            current_batch = tokens[i*batch_size: end_idx]+["[SEP]"]
            current_batch = ["[CLS]"] + current_batch if i > 0 else current_batch
        else:
            current_batch = tokens[i*batch_size: len(tokens)]
            currnet_batch = add_pad(current_batch)
                                            
        tokens_batches.append(current_batch)
     
    return tokens_batches
 
def make_attn_mask(tokens: list):
    attn_mask = [1 if token != "[PAD]" else 0 for token in tokens]
    return attn_mask

def encode_text(text: str,model,tokenizer):
    text = "%s %s %s" % ("[CLS]",text,"[SEP]")
    tokens = tokenizer.tokenize(text,add_special_tokens=True)
    tokens_batches = divide_tokens_into_batches(tokens)
    attn_mask_batches = [make_attn_mask(tokens) for tokens in tokens_batches]
    token_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_batches])
    attn_tensor = torch.tensor(attn_mask_batches,dtype=torch.long)
    output = model(token_tensor,attention_mask=attn_tensor)[0]
    
    embedding = torch.mean(output.select(1,0),0) if len(tokens_batches) > 0 else output
    return embedding
        
    

