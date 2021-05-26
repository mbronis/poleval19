import os
import pickle
import numpy as np

from fastapi import FastAPI

from src.models import SVM

from typing import Sequence
from src.data_types import Tweet, Tokens, Tag


app = FastAPI()

classifier = SVM()
classifier.load_model()
classifier.load_spacy_model()


@app.post("/tag_tokens", response_model=Sequence[Tag])
def predict(tokens: Sequence[Tokens]):
    response = classifier.tag_tokens(tokens)

    return response


@app.post("/tag_tweet", response_model=Tag)
def predict(tweet: Tweet):
    response = classifier.tag_tweet(tweet)

    return response
