from pydantic import BaseModel
from typing import Any, Dict, Tuple, Sequence

Tweet = str
Tweets = Sequence[Tweet]
Token = str
Tokens = Sequence[Token]
Tag = int
Tags = Sequence[Tag]

Score = Tuple[float, float]
Seconds = float


class TweetRequest(BaseModel):
    text: Tweet


class TaggedTweetResponse:
    tweet: Tweet
    tokenized: Tokens
    tag: Tag


class ModelPerformanceData:
    model_class: str
    model_name: str
    train_time: str
    model_params: Dict[str, Any]
    score_train: Score
    score_oof: Score
    score_test: Score
    train_duration: Seconds
    tag_duration: Seconds
    tags_train: Sequence[Tag]
    tags_train_pred: Sequence[Tag]
    tags_oof_pred: Sequence[Tag]
    tags_test: Sequence[Tag]
    tags_test_pred: Sequence[Tag]
