import pickle

import pandas as pd
import numpy as np
from IPython.display import display

from typing import List, Tuple, Dict, Sequence, Any

from src.data_model import Tokens, Tag, Score
from src.utils import Config, do_nothing

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier


class Classifier:
    """
    Base classifier class.

    Implements data transformers and utility functions.
    Adding final step in pipeline with chosen classifier algo makes inherit classes fully functional.

    Attributes
    ----------
    pipeline : sklearn.Pipeline
        Pipeline defining data transformations and final classifier.

    params : Dict[str, Any]
        Parameters for each pipeline step

    """
    tag = np.int64

    DEFAULT_PARAMS = {'vect__tokenizer': do_nothing, 'vect__preprocessor': None, 'vect__lowercase': False}

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        self.save_path = None
        self.model_name = None
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
            ])
        self.pipeline.set_params(**Classifier.DEFAULT_PARAMS)
        self.params = params

    @staticmethod
    def score(y_true: Sequence[Tag], y_pred: Sequence[Tag]) -> Score:
        """Return ``(f-micro, f-macro)``."""
        return f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro')

    @staticmethod
    def conf_matrix(y_true: Sequence[Tag], y_pred: Sequence[Tag]) -> pd.DataFrame:
        """Return multiclass confusion matrix"""

        df_raw = pd.DataFrame(zip(y_true, y_pred), columns=['act', 'pred'])
        df_gr = df_raw.groupby(['act', 'pred']).size().reset_index().rename({0: 'count'}, axis=1)
        conf_matrix = df_gr.pivot(index='act', columns='pred', values='count').fillna(0)

        return conf_matrix

    def make_tags(self, X: Sequence[Tokens]) -> Sequence[Tag]:
        return self.pipeline.predict(X)

    def train(self, X_train: Sequence[Tokens], y_train: Sequence[Tag]) -> Tuple[Sequence[Tag], Score]:
        """Train classifier, return train preds and score."""
        self.pipeline.fit(X_train, y_train)

        pred_tags = self.make_tags(X_train)
        pred_scores = self.score(y_train, pred_tags)

        print(pred_scores)

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
            tags_in_oof = self.make_tags(X_test)

            tags_train.iloc[train_index] = tags_in_train
            tags_oof.iloc[test_index] = tags_in_oof

        tags_train = tags_train.astype(np.int64)
        tags_oof = tags_oof.astype(np.int64)

        score_train = self.score(y, tags_train)
        score_oof = self.score(y, tags_oof)

        print(f"train: {score_train}, oof: {score_oof}")
        display(self.conf_matrix(y, tags_oof))

        return tags_oof, score_train, score_oof

    def save_model(self) -> None:
        path = self.save_path + self.model_name + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump((self.pipeline, self.params), f)

    def load_model(self) -> None:
        path = self.save_path + self.model_name + '.pkl'
        with open(path, 'rb') as f:
            self.pipeline, self.params = pickle.load(f)


class SVM(Classifier):
    def __init__(self, params: Dict[str, Any] = {}, model_name: str = 'default_svm') -> None:
        super().__init__(params)
        self.pipeline.steps.append(('clf', LinearSVC()))
        self.pipeline.set_params(**params)

        self.save_path = Config()['SVM']['model_path']
        self.model_name = model_name


class Ridge(Classifier):
    def __init__(self, params: Dict[str, Any] = {}, model_name: str = 'default_ridge') -> None:
        super().__init__(params)
        self.pipeline.steps.append(('clf', RidgeClassifier()))
        self.pipeline.set_params(**params)

        self.save_path = Config()['RIDGE']['model_path']
        self.model_name = model_name


class RF(Classifier):
    def __init__(self, params: Dict[str, Any] = {}, model_name: str = 'default_rf') -> None:
        super().__init__(params)
        self.pipeline.steps.append(('clf', RandomForestClassifier()))
        self.pipeline.set_params(**params)

        self.save_path = Config()['RF']['model_path']
        self.model_name = model_name
