import os
import pickle
import re
import spacy
import itertools

import pandas as pd
import numpy as np

from src.utils import Logger
from src.data_model import Tweet, Token, Tokens


class DataReader:
    """
    A class for reading and processing raw tweets.
    Processed data is stored in pd.DataFrame (``DataReader.df``)

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

    Attributes
    ----------
    df : pd.DataFrame
        Data frame with raw text and cleared tokens.
        Columns:
            Name: raw_tweets, dtype: str
            Name: clean_tweets, dtype: str
            Name: tokens, dtype: List[str]
            Name: tokens_count, dtype: int
            Name: tag, dtype: int
    """
    def __init__(self, text_file: str, tags_file: str = None, force_reload: bool = False) -> None:
        self._nlp = None
        self._logger = Logger('io')
        self.df = self._load_data(text_file, tags_file, force_reload)
        self._stats = None
        self.stats

    def _load_data(self, text_file: str, tags_file: str, force_reload: bool) -> pd.DataFrame:
        """
        Load dataframe with cleared and tokenized tweets.

        First tries to load processed data from pickle.
        If pickle not found, or ``force_reload`` is True, reads raw data and run processing.

        Parameters
        ----------
        text_file : str
            Name of a file with raw texts.
        tags_file : str
            Name of a file with tags.
        force_reload : bool
            If true loads from raw data even if pickle found.

        Returns
        -------
        pd.DataFrame
            Data frame with raw text and cleared tokens.
        """
        # TODO: to config
        pickle_path = './data/processed/'
        pickle_name = text_file.replace('.txt', '.pkl')

        if (pickle_name in os.listdir(pickle_path)) & ~force_reload:
            self._logger.log('reading from pickle')
            df = pickle.load(open(pickle_path+pickle_name, "rb"))
        else:
            self._logger.log('processing raw data')
            df = self._build_dataframe(text_file, tags_file)

        self._logger.log('data ready')
        return df

    def _build_dataframe(self, text_file: str, tags_file: str) -> pd.DataFrame:
        """
        Clear and tokenize raw texts.
        Pickle processed data

        Parameters
        ----------
        text_file : str
            Name of a file with raw texts.
        tags_file : str
            Name of a file with tags.

        Returns
        -------
        pd.DataFrame
            Data frame with raw text and cleared tokens.
        """
        # TODO: to config
        raw_path = './data/raw/'
        pickle_path = './data/processed/'
        pickle_name = text_file.replace('.txt', '.pkl')

        self._nlp = spacy.load('pl_spacy_model')

        with open(raw_path+text_file) as f:
            raw_tweets = f.readlines()

            tweets = [tweet.strip() for tweet in raw_tweets]
            tweets = [re.sub(r'@anonymized_account', '', tweet) for tweet in tweets]
            tweets = [re.sub(r'[^\w\s]', '', tweet) for tweet in tweets]
            tweets = [re.sub(r'[0-9]', '', tweet) for tweet in tweets]
            tweets = [re.sub(r' +', ' ', tweet) for tweet in tweets]
            tweets = [tweet.lower() for tweet in tweets]
            tweets = [tweet.strip() for tweet in tweets]

            df = pd.DataFrame(zip(raw_tweets, tweets), columns=['raw_tweets', 'clean_tweets'])

            df['tokens'] = df['clean_tweets'].apply(self._tokenize_tweet)
            df['tokens_count'] = df['tokens'].apply(len)
            if tags_file is not None:
                df['tag'] = pd.read_fwf(raw_path+tags_file, header=None)[0]
            else:
                df['tag'] = np.nan

            pickle.dump(df, open(pickle_path+pickle_name, "wb"))
            return df

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

    def _tokenize_tweet(self, tweet: Tweet) -> Tokens:
        """Return list of cleared tokens with length > 1."""
        tokens = [tok.lemma_ for tok in self._nlp(tweet)]
        tokens = [DataReader._latinize_diacritics(tok) for tok in tokens]
        tokens = [DataReader._drop_adjacent_equals(tok) for tok in tokens]
        tokens = [DataReader._collapse_exploded(tok) for tok in tokens]
        tokens = [tok for tok in tokens if len(tok) > 1]

        return tokens

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
