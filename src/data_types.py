from pydantic import BaseModel
from typing import Tuple, Sequence

Tweet = str
Tweets = Sequence[Tweet]
Token = str
Tokens = Sequence[Token]
Tag = int
Tags = Sequence[Tag]

Score = Tuple[float, float]


class TweetRequest(BaseModel):
    text: Tweet


class TaggedTweetResponse:
    tweet: Tweet
    tokenized: Tokens
    tag: Tag
