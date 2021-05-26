
from typing import List, Tuple, Sequence

Tweet = str
Token = str
Tokens = Sequence[Token]
Tag = int
Score = Tuple[float, float]


class TaggedTweetResponse:
    tweet: Tweet
    tokenized: Tokens
    tag: Tag
