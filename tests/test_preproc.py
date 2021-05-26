from doctest import DocTestSuite
from unittest import TestCase, TestSuite, TestLoader, TextTestRunner

from src.preproc import DataReader, Preprocessor


class TestDataReader(TestCase):

    def test_df_shape(self):
        tweets = './data/raw/test_tweets.txt'
        tags = './data/raw/test_tags.txt'

        dr = DataReader(tweets, tags)

        self.assertEqual(dr.df.shape, (54, 4))

    def test_df_columns(self):
        tweets = './data/raw/test_tweets.txt'
        tags = './data/raw/test_tags.txt'

        dr = DataReader(tweets, tags)

        self.assertEqual(list(dr.df.columns), ['raw_tweets', 'tokens', 'tokens_count', 'tag'])


class TestPreprocessor(TestCase):

    def test_base_cleanup(self):
        pr = Preprocessor()

        test1 = (' @anonymized_account$@#\\  pan123 t.o JEST', 'pan to jest')

        self.assertEqual(pr._base_cleanup(test1[0]), test1[1])

    def test_tokenizer(self):
        pr = Preprocessor()

        test1 = ('mam to coś', ['mieć', 'ten', 'coś'])

        self.assertEqual(pr._tokenizer(test1[0]), test1[1])

    def test_drop_adjacent_equals(self):
        pr = Preprocessor()

        test1 = ('kkk', 'k')
        test2 = ('lekkie pióórko', 'lekie piórko')

        self.assertEqual(pr._drop_adjacent_equals(test1[0]), test1[1])
        self.assertEqual(pr._drop_adjacent_equals(test2[0]), test2[1])

    def test_drop_adjacent_equals(self):
        pr = Preprocessor()

        test1 = ('jesteś b r z y d k i', 'jesteś brzydki')

        self.assertEqual(pr._collapse_exploded(test1[0]), test1[1])

    def test_latinize_diacritics(self):
        pr = Preprocessor()

        test1 = ('gęśl', 'gesl')

        self.assertEqual(pr._latinize_diacritics(test1[0]), test1[1])
