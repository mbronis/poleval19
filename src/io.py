import os
import pickle
import re
import spacy
import itertools

import pandas as pd
import numpy as np

from typing import List
from src.utils import Logger


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
        self._nlp = spacy.load('pl_spacy_model')
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
    def _drop_adjacent_equals(word: str) -> str:
        """
        Remove adjacent duplicate characters.

        Examples
        --------
        >>> _drop_adjacent_equals('kkk')
        'k'

        >>> _drop_adjacent_equals('lekkie pióórko')
        'lekie piórko'
        """
        return ''.join(c[0] for c in itertools.groupby(word))

    @staticmethod
    def _collapse_exploded(word: str, separators: str = ' .-_') -> str:
        """
        Collapse word expanded with ``separators``.

        Example
        --------
        >>> _collapse_exploded('jesteś b r z y d k i')
        'jesteś brzydki'
        """
        if len(word) < 5:
            return word

        remove = []
        for i, l in enumerate(word[2:-1]):
            if l in separators:
                if (word[i - 2] in separators) & (word[i + 2] in separators):
                    if (word[i - 1].isalpha()) & (word[i + 1].isalpha()):
                        remove.append(i)
                        remove.append(i + 2)

        return ''.join([l for i, l in enumerate(word) if i not in remove])

    @staticmethod
    def _latinize_diacritics(word: str) -> str:
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
        return word.translate(table)

    def _tokenize_tweet(self, tweet: str) -> List[str]:
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

# class Dataset:
#     def __init__(self, texts_file, tags_file, clean_data=True, remove_stopwords=False, is_train=True):
#         self.args = Parser().get_sections(['GENERAL', 'RNN', 'FLAIR'])
#         self.max_sent_length = int(self.args['max_sent_length'])
#         self.batch_size = int(self.args['batch_size'])
#         self.emb_size = int(self.args['emb_size'])
#         self.clean_data = clean_data
#         self.remove_stopwords = remove_stopwords
#         self.is_train = is_train
#
#         self.nlp = Polish()
#         self.df = self.build_dataframe(texts_file, tags_file)
#         self.unk_emb = self.get_random_emb(self.emb_size)
#         self.word2idx, self.idx2word = self.build_dict()
#         if self.is_train:
#             self.embeddings = self.get_embeddings(self.args['emb_path'])
#
#     def build_dataframe(self, texts_file, tags_file):
#         with open(texts_file) as file:
#             lines = [line.strip() for line in file.readlines()]
#             texts = pd.DataFrame(lines, columns=['text'])
#         tags = pd.read_fwf(tags_file, header=None, names=['tag'])
#         df = pd.concat([texts, tags], axis=1)
#         df['tokens'] = df['text'].map(lambda x: self.preprocess_sentence(x))
#         df['length'] = df['tokens'].map(lambda x: len(x))
#         df['clean_text'] = df['tokens'].map(lambda x: " ".join(x))
#         if self.clean_data:
#             df = self.clean(df)
#         return df
#
#     def preprocess_sentence(self, sentence):
#         sentence = sentence.replace(r"\n", "").replace(r"\r", "").replace(r"\t", "").replace("„", "").replace("”", "")
#         doc = [tok for tok in self.nlp(sentence)]
#         if not self.clean_data and str(doc[0]) == "RT":
#             doc.pop(0)
#         while str(doc[0]) == "@anonymized_account":
#             doc.pop(0)
#         while str(doc[-1]) == "@anonymized_account":
#             doc.pop()
#         if self.remove_stopwords:
#             doc = [tok for tok in doc if not tok.is_stop]
#         doc = [tok.lower_ for tok in doc]
#         doc = ["".join(c for c in tok if not c.isdigit() and c not in string.punctuation) for tok in doc]
#         doc = [RE_EMOJI.sub(r'', tok) for tok in doc]
#         doc = [tok.strip() for tok in doc if tok.strip()]
#         return doc
#
#     def build_dict(self):
#         if self.is_train:
#             sentences = self.df['tokens']
#             all_tokens = [token for sentence in sentences for token in sentence]
#             words_counter = Counter(all_tokens).most_common()
#             word2idx = {
#                 self.args['pad']: 0,
#                 self.args['unk']: 1
#             }
#             for word, _ in words_counter:
#                 word2idx[word] = len(word2idx)
#
#             with open(self.args['word_dict_path'], 'wb') as dict_file:
#                 pickle.dump(word2idx, dict_file)
#
#         else:
#             with open(self.args['word_dict_path'], 'rb') as dict_file:
#                 word2idx = pickle.load(dict_file)
#
#         idx2word = {idx: word for word, idx in word2idx.items()}
#         return word2idx, idx2word
#
#     def transform_dataset(self):
#         sentences = self.df['tokens'].values
#         x = [sentence[:self.max_sent_length] for sentence in sentences]
#         x = [sentence + [self.args['pad']] * (self.max_sent_length - len(sentence)) for sentence in x]
#         x = [[self.word2idx.get(word, self.word2idx[self.args['unk']]) for word in sentence] for sentence in x]
#         y = self.df['tag'].values
#         return np.array(x), np.array(y)
#
#     def parse_dataset(self):
#         x, y = self.transform_dataset()
#         if self.is_train:
#             x, y = shuffle(x, y, random_state=42)
#             train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)
#             return list(self.chunks(train_x, train_y, self.batch_size)), valid_x, valid_y
#         return list(self.chunks(x, y, self.batch_size))
#
#     def get_embeddings(self, embeddings_file):
#         emb_list = []
#         print("Loading vectors...")
#         word_vectors = KeyedVectors.load_word2vec_format(embeddings_file, binary=False)
#         print("Vectors loaded...")
#         for _, word in sorted(self.idx2word.items()):
#             if word == self.args['pad']:
#                 word_vec = np.zeros(self.emb_size)
#             elif word == self.args['unk']:
#                 word_vec = self.unk_emb
#             else:
#                 try:
#                     word_vec = word_vectors.word_vec(word)
#                 except KeyError:
#                     word_vec = self.unk_emb
#             emb_list.append(word_vec)
#         return np.array(emb_list, dtype=np.float32)
#
#     def get_class_weight(self):
#         y = self.df['tag'].values
#         _, counts = np.unique(y, return_counts=True)
#         return np.array(1 - counts / y.size)
#
#     def prepare_flair_format(self, column_name):
#         X = self.df[column_name].values
#         y = self.df['tag'].values
#
#         train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
#         train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_y, test_size=0.15, random_state=42,
#                                                           stratify=train_y)
#         train_data = [f"__label__{label} {text}" for text, label in zip(train_x, train_y)]
#         val_data = [f"__label__{label} {text}" for text, label in zip(val_x, val_y)]
#         dev_data = [f"__label__{label} {text}" for text, label in zip(dev_x, dev_y)]
#         flair_dir_path = self.args['flair_data_path']
#         os.makedirs(flair_dir_path, exist_ok=True)
#
#         with open(os.path.join(flair_dir_path, f"train_{column_name}.txt"), 'w') as train_file, \
#                 open(os.path.join(flair_dir_path, f"dev_{column_name}.txt"), 'w') as dev_file, \
#                 open(os.path.join(flair_dir_path, f"test_{column_name}.txt"), 'w') as val_file:
#             train_file.write("\n".join(train_data))
#             dev_file.write("\n".join(dev_data))
#             val_file.write("\n".join(val_data))
#
#     def print_stats(self):
#         print(self.df['length'].describe())
#         print(self.df['length'].quantile(0.95, interpolation='lower'))
#         print(self.df['length'].quantile(0.99, interpolation='lower'))
#         print(self.df.shape)
#         print(self.df['tag'].value_counts())
#
#     @staticmethod
#     def get_random_emb(length):
#         return np.random.uniform(-0.25, 0.25, length)
#
#     @staticmethod
#     def clean(dataframe):
#         dataframe = dataframe.drop_duplicates('clean_text')
#         return dataframe[(dataframe['tokens'].apply(lambda x: "rt" not in x[:1])) & (dataframe['length'] > 1)]
#
#     @staticmethod
#     def chunks(inputs, outputs, batch_size):
#         for i in range(0, len(inputs), batch_size):
#             yield inputs[i:i + batch_size], outputs[i:i + batch_size]
#
#
# if __name__ == "__main__":
#     dt = Dataset("../data/task_6-2/training_set_clean_only_text.txt",
#                  "../data/task_6-2/training_set_clean_only_tags.txt",
#                  True, False, False)
#     # dt.prepare_flair_format("clean_text")
