import pickle

import numpy as np

from typing import List, Tuple, Dict, Sequence, Any

from src.data_types import Tweet, Tokens, Tag, Score
from src.utils import Config, do_nothing
from src.preproc import Preprocessor
from src.model_utils import score

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier


class Classifier:
    """
    Base classifier class.

    Implements data transformers and utility functions.
    Adding final step in pipeline with chosen classifier algo makes inherit classes fully functional.

    Attributes
    ----------
    pipeline : sklearn.Pipeline
        Pipeline defining data transformations and final classifier.

    pipe_params : Dict[str, Any]
        Parameters for each pipeline step

    """
    DEFAULT_PARAMS = {'vect__tokenizer': do_nothing, 'vect__preprocessor': None, 'vect__lowercase': False}

    def __init__(self, pipe_params: Dict[str, Any] = {}) -> None:
        self.save_path = None
        self.model_name = None
        self.preprocessor = Preprocessor()
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
            ])
        self.pipeline.set_params(**Classifier.DEFAULT_PARAMS)
        self.pipe_params = pipe_params

    def tag_tokens(self, tokens: Sequence[Tokens]) -> List[Tag]:
        """Tag preprocessed tweets"""

        tags = [tag for tag in self.pipeline.predict(tokens)]

        return tags

    def tag_tweet(self, tweet: Tweet) -> Tag:
        tokens: Tokens = self.preprocessor.transform_tweet(tweet)
        tag = self.pipeline.predict([tokens])[0]

        return tag

    def train(self, tokens: Sequence[Tokens], tags: Sequence[Tag]) -> Tuple[Sequence[Tag], Score]:
        """Train classifier, return train preds and score."""
        self.pipeline.fit(tokens, tags)

        pred_tags = self.tag_tokens(tokens)
        pred_scores = score(tags, pred_tags)

        return pred_tags, pred_scores

    def score_cv(self, X: Sequence[Tokens], y: Sequence[Tag],
                 folds: int = 5, seed: int = 42) -> Tuple[Sequence[Tag], Score, Score]:
        """Return oof preds and ``(train, oof)`` scores."""

        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        kfold.get_n_splits(X, y)
        splits = kfold.split(X, y)

        tags_train = y.apply(lambda x: np.nan)
        tags_oof = y.apply(lambda x: np.nan)

        for train_index, test_index in splits:
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            tags_in_train, _ = self.train(X_train, y_train)
            tags_in_oof = self.tag_tokens(X_test)

            tags_train.iloc[train_index] = tags_in_train
            tags_oof.iloc[test_index] = tags_in_oof

        tags_train = tags_train.astype(np.int64)
        tags_oof = tags_oof.astype(np.int64)

        score_train = score(y, tags_train)
        score_oof = score(y, tags_oof)

        return tags_oof, score_train, score_oof

    def save_model(self) -> None:
        path = self.save_path + self.model_name + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump((self.pipeline, self.pipe_params), f)

    def load_model(self) -> None:
        path = self.save_path + self.model_name + '.pkl'
        with open(path, 'rb') as f:
            self.pipeline, self.pipe_params = pickle.load(f)

    def load_spacy_model(self):
        if self.preprocessor is not None:
            self.preprocessor.load_spacy_model()


class FlatPredictor(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X, y=None):
        return [0 for _ in X]


class Baseline(Classifier):
    def __init__(self, pipe_params: Dict[str, Any] = {}, model_name: str = 'baseline') -> None:
        super().__init__(pipe_params)
        self.pipeline.steps.append(('clf', FlatPredictor()))
        self.pipeline.set_params(**pipe_params)

        self.save_path = Config()['Baseline']['model_path']
        self.model_name = model_name


class SVM(Classifier):
    def __init__(self, pipe_params: Dict[str, Any] = {}, model_name: str = 'default_svm') -> None:
        super().__init__(pipe_params)
        self.pipeline.steps.append(('clf', LinearSVC()))
        self.pipeline.set_params(**pipe_params)

        self.save_path = Config()['SVM']['model_path']
        self.model_name = model_name


class Ridge(Classifier):
    def __init__(self, pipe_params: Dict[str, Any] = {}, model_name: str = 'default_ridge') -> None:
        super().__init__(pipe_params)
        self.pipeline.steps.append(('clf', RidgeClassifier()))
        self.pipeline.set_params(**pipe_params)

        self.save_path = Config()['RIDGE']['model_path']
        self.model_name = model_name


class RF(Classifier):
    def __init__(self, pipe_params: Dict[str, Any] = {}, model_name: str = 'default_rf') -> None:
        super().__init__(pipe_params)
        self.pipeline.steps.append(('clf', RandomForestClassifier()))
        self.pipeline.set_params(**pipe_params)

        self.save_path = Config()['RF']['model_path']
        self.model_name = model_name


class LGBM(Classifier):
    def __init__(self, pipe_params: Dict[str, Any] = {}, model_name: str = 'default_gbm') -> None:
        super().__init__(pipe_params)
        self.pipeline.steps.append(('clf', LGBMClassifier()))
        self.pipeline.set_params(**pipe_params)

        self.save_path = Config()['LGBM']['model_path']
        self.model_name = model_name
