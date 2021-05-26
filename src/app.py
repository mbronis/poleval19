import os
import pickle
import numpy as np

from fastapi import FastAPI

from src.models import SVM

from typing import Sequence
from src.data_types import Tokens, Tag


app = FastAPI()


@app.post("/predict", response_model=Sequence[Tag])
def predict(tweet: Sequence[Tokens]):
    classifier = SVM()
    classifier.load_model()

    response = classifier.make_tags(tweet)
    response = list(response)

    return response
