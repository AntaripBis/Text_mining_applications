import argparse
from pathlib import Path
import requests
import urllib3
import os
import shutil

from zipfile import ZipFile
from tqdm import tqdm

DOWNLOAD_SERVER_URL = 'https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/'

PRETRAINED_MODEL_URL_MAP = {
    "roberta-base-nli-stsb-mean-tokens": "%s%s%s" % (DOWNLOAD_SERVER_URL,"roberta-base-nli-stsb-mean-tokens",".zip"),
    "roberta-large-nli-stsb-mean-tokens": "%s%s%s" % (DOWNLOAD_SERVER_URL,"roberta-large-nli-stsb-mean-tokens",".zip"),
    "distilbert-base-nli-stsb-mean-tokens": "%s%s%s" % (DOWNLOAD_SERVER_URL,"distilbert-base-nli-stsb-mean-tokens",".zip")
}

def http_get(url, path):
    with open(path, "wb") as file_binary:
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
            req.raise_for_status()

        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)
    progress.close()
    
def download_pretrained_files(model_name, location):
    model_loc_dir = model_name.replace("-","_")
    location = os.path.join(location,model_loc_dir)
    if not os.path.exists(location):
        os.makedirs(location,exist_ok=True)
    target_path = os.path.join(location,"model.zip")
    
    try:
        model_url = PRETRAINED_MODEL_URL_MAP[model_name]
        print("Model download URL is %s" % (model_url))

        http_get(model_url, target_path)
        with ZipFile(target_path, 'r') as zip:
             zip.extractall(location)
        print("Model %s has been successfully downloaded" % (model_name))
    except Exception as e:
        shutil.rmtree(location)
        raise e
    
        
def read_args():
    parser = argparse.ArgumentParser(description="arguments required to download pretrained models")
    parser.add_argument("--models","-m",dest="models",default=None,type=str,nargs="*",help="Model names that are to be downloaded")
    parser.add_argument("--location","-l",dest="location",default="pretrained_models",type=str,help="Location where model is to be downloaded")
    return parser

def main():
    parser = read_args()
    args = parser.parse_args()
    assert args.models is not None and len(args.models) > 0, "Atleast one model name needs to be specified"
    print(args)
    [download_pretrained_files(model_name,args.location)for model_name in args.models if model_name is not None]
    
    
if __name__=="__main__":
    main()