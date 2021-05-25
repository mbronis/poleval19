import os
import pickle
import numpy as np

from fastapi import FastAPI

from src.models import SVM

from typing import List, Sequence


app = FastAPI()
tokens = List[str]
tag = int


@app.post("/predict", response_model=Sequence[tag])
def predict(tweet: Sequence[tokens]):
    classifier = SVM()
    classifier.load_model()

    response = classifier.make_tags(tweet)
    response = list(response)

    return response
