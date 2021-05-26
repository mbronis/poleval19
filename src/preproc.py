from __future__ import annotations

import os
import pickle
import re
import spacy
import itertools

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from src.utils import Logger
from typing import List
from src.data_types import Tweets, Tweet, Token, Tokens, Tags


class Preprocessor(BaseEstimator):
    """
    Class for cleaning and tokenizing tweet's raw text

    Steps:
        1. remove ``@anonymized_account`` tag
        2. remove chars other than letters and spaces
        3. remove duplicate spaces
        4. apply lowercase
        5. lemmatizes tokens with ``pl_spacy_model``
        6. convert polish diacritics to latin letters
        7. drop adjacent equals letters
        8. collapse words exploded with spaces
        9. remove zero/one letter tokens
    """

    def __init__(self, min_tok_len: int = 2):
        self._min_tok_len = min_tok_len
        self._logger = Logger('preproc')
        self._nlp = None

    def fit(self, tweets: Tweets, tags: Tags = None) -> Preprocessor:
        return self

    def transform_tweet(self, tweet: Tweet) -> Tokens:

        tweet: Tweet = self._base_cleanup(tweet)
        tokens: Tokens = self._tokenizer(tweet)
        tokens = [Preprocessor._latinize_diacritics(tok) for tok in tokens]
        tokens = [Preprocessor._drop_adjacent_equals(tok) for tok in tokens]
        tokens = [Preprocessor._collapse_exploded(tok) for tok in tokens]
        tokens = [tok for tok in tokens if len(tok) >= self._min_tok_len]

        return tokens

    def transform(self, tweets: Tweets, tags: Tags = None) -> List[Tokens]:
        tokens = [self.transform_tweet(tweet) for tweet in tweets]

        return tokens

    @staticmethod
    def _base_cleanup(tweet: Tweet) -> Tweet:
        """Keep only letters and spaces, apply to lower, remove ``@anonymized_account`` and extra spaces"""
        tweet = tweet.strip()
        tweet = re.sub(r'@anonymized_account', '', tweet)
        tweet = re.sub(r'[^\w\s]', '', tweet)
        tweet = re.sub(r'[0-9]', '', tweet)
        tweet = re.sub(r' +', ' ', tweet)
        tweet = tweet.lower()
        tweet = tweet.strip()

        return tweet

    def _tokenizer(self, tweet: Tweet) -> Tokens:
        """Tokenize tweet"""
        if self._nlp is None:
            self._nlp = spacy.load('pl_spacy_model')
        tokens = [tok.lemma_ for tok in self._nlp(tweet)]

        return tokens

    @staticmethod
    def _drop_adjacent_equals(tok: Token) -> Token:
        """
        Remove adjacent duplicate characters.

        Examples
        --------
        >>> _drop_adjacent_equals('kkk')
        'k'

        >>> _drop_adjacent_equals('lekkie pióórko')
        'lekie piórko'
        """
        return ''.join(c[0] for c in itertools.groupby(tok))

    @staticmethod
    def _collapse_exploded(tok: Token, separators: str = ' .-_') -> Token:
        """
        Collapse word expanded with ``separators``.

        Example
        --------
        >>> _collapse_exploded('jesteś b r z y d k i')
        'jesteś brzydki'
        """
        if len(tok) < 5:
            return tok

        remove = []
        for i, l in enumerate(tok[2:-1]):
            if l in separators:
                if (tok[i - 2] in separators) & (tok[i + 2] in separators):
                    if (tok[i - 1].isalpha()) & (tok[i + 1].isalpha()):
                        remove.append(i)
                        remove.append(i + 2)

        return ''.join([l for i, l in enumerate(tok) if i not in remove])

    @staticmethod
    def _latinize_diacritics(tok: Token) -> Token:
        """
        Convert polish diacritics to latin letters.

        Example
        --------
        >>> _latinize_diacritics('gęśl')
        'gesl'
        """
        letters_diac = 'ąćęłńóśżźĄĆĘŁŃÓŚŻŹ'
        letters_latin = 'acelnoszzACELNOSZZ'
        table = str.maketrans(letters_diac, letters_latin)
        return tok.translate(table)


class DataReader:
    """
    Class for loading and processing raw tweets.

    Attributes
    ----------
    df : pd.DataFrame
        Data frame with raw text and cleared tokens.
        Columns:
            Name: raw_tweets, dtype: str
            Name: tokens, dtype: List[str]
            Name: tokens_count, dtype: int
            Name: tag, dtype: int
    """
    def __init__(self, text_file: str, tags_file: str = None, force_reload: bool = False) -> None:
        self._logger = Logger('io')
        self._preprocessor = Preprocessor()
        self.df = self._load_data(text_file, tags_file, force_reload)
        self._stats = None
        self.stats

    def _load_data(self, tweets_path: str, tags_path: str, force_reload: bool = False) -> pd.DataFrame:
        """
        Load dataframe with cleared and tokenized tweets.

        First tries to load processed data from pickle.
        If pickle not found, or ``force_reload`` is True, reads raw data and run processing.

        Parameters
        ----------
        tweets_path : str
            Name of a file with raw texts.
        tags_path : str
            Name of a file with tags.
        force_reload : bool
            If true loads from raw data even if pickle found.

        Returns
        -------
        pd.DataFrame
            Data frame with raw text and cleared tokens.
        """
        pickle_path = tweets_path.replace('.txt', '.pkl').replace('raw', 'processed')
        pickle_folder, pickle_name = os.path.split(pickle_path)

        if (pickle_name in os.listdir(pickle_folder)) & ~force_reload:
            self._logger.log('reading from pickle')
            with open(pickle_path, "rb") as f:
                df = pickle.load(f)
        else:
            self._logger.log('processing raw data')
            df = self._build_dataframe(tweets_path, tags_path)

        self._logger.log('data ready')
        return df

    def _build_dataframe(self, tweets_path: str, tags_path: str) -> pd.DataFrame:
        """
        Clear and tokenize raw texts.
        Pickle processed data

        Parameters
        ----------
        tweets_path : str
            Name of a file with raw texts.
        tags_path : str
            Name of a file with tags.

        Returns
        -------
        pd.DataFrame
            Data frame with raw text and cleared tokens.
        """
        with open(tweets_path) as f:
            raw_tweets = f.readlines()

            df = pd.DataFrame(raw_tweets, columns=['raw_tweets'])
            df['tokens'] = self._preprocessor.transform(raw_tweets)
            df['tokens_count'] = df['tokens'].apply(len)

            if tags_path is not None:
                df['tag'] = pd.read_fwf(tags_path, header=None)[0]
            else:
                df['tag'] = np.nan

            pickle_path = tweets_path.replace('.txt', '.pkl').replace('raw', 'processed')
            with open(pickle_path, "wb") as p:
                pickle.dump(df, p)

            return df

    @property
    def stats(self):
        self._stats = dict()
        self._stats['tweets count'] = self.df.shape[0]
        self._stats['tokens in tweet distribution'] = self.df['tokens_count'].describe([.25, .5, .75, .95, .99])
        self._stats['unique tokens'] = len({toc for tweet_toc in self.df['tokens'] for toc in tweet_toc})
        self._stats['tags count'] = self.df['tag'].value_counts().sort_index()

        print("-------- stats --------")
        for stat, value in self._stats.items():
            print(f"=======================\n{stat}:\n{value}")
