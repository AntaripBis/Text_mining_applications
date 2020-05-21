import os
import json
import pickle
import sys
import signal
import traceback
import re
import flask

import torch
from sentence_transformers import SentenceTransformer

from pathlib import Path

import warnings

from .text_utils import load_spacy
from .text_embedding import encode_text_with_sent_bert

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

prefix = "/opt/ml/"

PATH = os.path.join(prefix, "model")

MODEL_PATH = os.path.join(PATH, "roberta_base_nli_stsb_mean_tokens")

# request_text = None


class EmbeddingService(object):
    model = None  # Where we keep the model when it's loaded
    spcy_eng = None

    @staticmethod
    def init_predictor_model():

        # Get model predictor
        EmbeddingService.model = SentenceTransformer(MODEL_PATH,device="gpu") if EmbeddingService.model is None else EmbeddingService.model
        EmbeddingService.spacy_eng = load_spacy() if EmbeddingService.spacy_eng is None else EmbeddingService.spacy_eng

        return True

    @staticmethod
    def embed_text(text,is_call_transcript: bool=True):
        embedding = encode_text_with_sent_bert(text,EmbeddingService.spacy_eng,EmbeddingService.model,
                                               is_call_transcript=is_call_transcript)

        return embedding



# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        EmbeddingService.model is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    is_call_transcript = False
    text = None

    if flask.request.content_type == "application/json":
#         print("calling json launched")
        data = flask.request.get_json(silent=True)

        text = data["text"]
        is_call_transcript = is_call_transcript if "is_call_transcript" not in data else data['is_call_transcript']

    else:
        return flask.Response(
            response="This predictor only supports JSON data",
            status=415,
            mimetype="text/plain",
        )

#     print("Invoked with text: {}.".format(text.encode("utf-8")))

    # Do the prediction
    embedding = EmbeddingService.embed(text, is_call_transcript)

    result = json.dumps({"text": text, "embedding": embedding})

    return flask.Response(response=result, status=200, mimetype="application/json")