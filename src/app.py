from typing import Sequence
from src.data_types import TweetRequest, Tokens, Tag

from fastapi import FastAPI

from src.models import RF, Ridge


app = FastAPI()

classifier_a = RF(model_name='final_rf')
classifier_a.load_model()
classifier_a.load_spacy_model()

classifier_b = Ridge(model_name='final_ridge')
classifier_b.load_model()
classifier_b.load_spacy_model()


@app.post("/tag_tokens", response_model=Sequence[Tag])
def predict(tokens: Sequence[Tokens]):
    response_a = classifier_a.tag_tokens(tokens)
    response_b = classifier_b.tag_tokens(tokens)

    response = [1 if 1 in x else (2 if 2 in x else 0) for x in zip(response_a, response_b)]

    return response


@app.post("/tag_tweet", response_model=Tag)
def predict(tweet: TweetRequest):
    text = tweet.text
    response_a = classifier_a.tag_tweet(text)
    response_b = classifier_b.tag_tweet(text)
    response = [response_a, response_b]

    return 1 if 1 in response else 2 if 2 in response else 0
