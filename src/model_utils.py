import ast
import numpy as np
import pandas as pd
from datetime import datetime
from random import choice

from typing import Any, Sequence, Dict
from src.data_types import Tokens, Tag, Score, ModelPerformanceData

from src.utils import Config
from sklearn.metrics import f1_score

from IPython.display import display

from functools import partial
from hyperopt import (
    fmin,
    tpe,
    mix,
    rand,
    Trials,
    STATUS_OK,
    STATUS_FAIL,
)


def score(y_true: Sequence[Tag], y_pred: Sequence[Tag], prec: int = 4) -> Score:
    """Return ``(f-micro, f-macro)``."""
    fmicro = round(f1_score(y_true, y_pred, average='micro'), prec)
    fmacro = round(f1_score(y_true, y_pred, average='macro'), prec)

    return fmicro, fmacro


def conf_matrix(y_true: Sequence[Tag], y_pred: Sequence[Tag]) -> pd.DataFrame:
    """Return multiclass confusion matrix"""

    df_raw = pd.DataFrame(zip(y_true, y_pred), columns=['act', 'pred'])
    df_gr = df_raw.groupby(['act', 'pred']).size().reset_index().rename({0: 'count'}, axis=1)
    matrix = df_gr.pivot(index='act', columns='pred', values='count').fillna(0)

    return matrix


def model_performance(model: Any,
                      train_tokens: Sequence[Tokens], train_tags: Sequence[Tag],
                      test_tokens: Sequence[Tokens], test_tags: Sequence[Tag],
                      save_model: bool = False
                      ) -> ModelPerformanceData:
    """Returns model train and validation statistics"""

    train_start = datetime.utcnow()
    train_tags_pred, score_train = model.train(train_tokens, train_tags)
    train_duration = round((datetime.utcnow() - train_start).total_seconds(), 2)

    if save_model:
        model.save_model()

    tag_start = datetime.utcnow()
    _ = model.tag_tokens(train_tokens)
    tag_duration = round((datetime.utcnow() - tag_start).total_seconds(), 2)

    test_tags_pred = model.tag_tokens(test_tokens)
    score_test = score(test_tags, test_tags_pred)

    oof_tags_pred, _, score_oof = model.score_cv(train_tokens, train_tags)

    results = ModelPerformanceData()
    results.model_class = type(model).__name__
    results.model_name = model.model_name
    results.model_params = model.pipe_params
    results.train_time = str(train_start)
    results.train_duration = train_duration
    results.tag_duration = tag_duration
    results.score_train = score_train
    results.score_oof = score_oof
    results.score_test = score_test
    results.tags_train = train_tags
    results.tags_train_pred = train_tags_pred
    results.tags_oof_pred = oof_tags_pred
    results.tags_test = test_tags
    results.tags_test_pred = test_tags_pred

    return results


def model_performance_summary(results: ModelPerformanceData,
                              scores: bool = True,
                              matrix: bool = False
                              ) -> None:
    """Prints model train and validation statistics"""

    if scores:
        print(f"\nModel: {results.model_name} ({results.model_class})")

        print("                macro\tmicro")
        print(f"score train:\t{results.score_train[0]}\t{results.score_train[1]}")
        print(f"score oof:\t{results.score_oof[0]}\t{results.score_oof[1]}")
        print(f"score test:\t{results.score_test[0]}\t{results.score_test[1]}")

    if matrix:
        print(f"\n------------------------------------------------------")
        print(f"Model: {results.model_name} ({results.model_class})")

        print("\ntrain conf matrix")
        display(conf_matrix(results.tags_train, results.tags_train_pred))

        print("\noof conf matrix")
        display(conf_matrix(results.tags_train, results.tags_oof_pred))

        print("\ntest conf matrix")
        display(conf_matrix(results.tags_test, results.tags_test_pred))


class Optimizer:
    """
    Class for optimizing hyper-parameters.

    Wrapper on hyperopt with added save of results for each point in hyperspace evaluated.

    Attributes
    ----------
    results : pd.DataFrame
        Data Frame with details of each hyperspace point evaluated.
        Columns:
            status: hyperopt.status: flag is hyper point evaluation succeeded
            loss: Float: value of loss (lower is better)
            score_train: Float: f1 macro on train set
            score_oof: Float: f1 macro on oof set
            score_diff: Float: difference in oof and train score
            params: Dict[str, Any]: params defining hyperpoint


    Methods
    ----------
    optimize: runs hyperopt optimization with tpe and rand mixture

    """

    # fixes hp.quniform returning float bug
    INT_PARAMS = [
        'clf__max_iter',
        'clf__n_estimators',
        'clf__max_depth',
        'clf__min_samples_leaf',
        'clf__max_leaf_nodes',
        'clf__n_jobs',
        'clf__num_iterations',
        'clf__num_leaves',
        'clf__min_data_in_leaf',
    ]

    def __init__(self):
        self.results = None
        self.penalty = float(Config()['HYPER']['penalty'])

    @staticmethod
    def _results_placeholder() -> pd.DataFrame:
        return pd.DataFrame(columns=['status', 'loss', 'score_train', 'score_oof', 'score_diff', 'params'])

    def _results_cleanup(self) -> None:
        self.results = self.results.sort_values('loss').reset_index(drop=True)

    def optimize(self,
                 model_class: Any,
                 train_tokens: Sequence[Tokens], train_tags: Sequence[Tag],
                 param_space,
                 int_params=INT_PARAMS,
                 n_rounds=10
                 ) -> None:
        """
        Runs hyper parameters space search.

        Updates self.results Data Frame with details of each hyperspace point evaluated.

        Parameters:
            model_class (Any): classifier class
            train_tokens (Sequence[Tokens]): tokens to train on
            train_tags (Sequence[Tag]): labels to train on
            param_space (Dict[str, hyperopt.hp): hyper-parameters space
            int_params (List[str]): list of integer based params
            n_rounds (int): number of hyperopt trials

        Returns:
           None
        """

        self.results = Optimizer._results_placeholder()

        def _fix_int_params(params):
            for p in params.keys():
                if p in int_params:
                    params[p] = int(params[p])

            return params

        def _f1_macro_with_penalty_loss(params):
            params = _fix_int_params(params)
            try:
                model = model_class(params)
                _, score_train = model.train(train_tokens, train_tags)
                _, _, score_oof = model.score_cv(train_tokens, train_tags)
            except Exception:
                return {
                    'status': STATUS_FAIL,
                    'loss': None,
                    'auc_train': None,
                    'auc_oof': None,
                    'auc_oot': None,
                    'params': str(params),
                }

            score_diff = score_oof[0]-score_train[0]
            loss_value = -(score_oof[0] + self.penalty * score_diff)

            result = {
                'status': STATUS_OK,
                'loss': loss_value,
                'score_train': score_train[0],
                'score_oof': score_oof[0],
                'score_diff': score_diff,
                'params': str(params),
            }
            self.results = pd.concat([self.results, pd.DataFrame(result, index=[0])], ignore_index=True)

            return result

        _ = fmin(
            fn=_f1_macro_with_penalty_loss,
            space=param_space,
            algo=partial(mix.suggest, p_suggest=[
                (.2, rand.suggest),
                (.8, tpe.suggest),
            ]),
            max_evals=n_rounds,
            trials=Trials(),
            max_queue_len=10,
            show_progressbar=True,
        )

        self._results_cleanup()


def median_cv_score(model_class: Any,
                    train_tokens: Sequence[Tokens], train_tags: Sequence[Tag],
                    params: Dict[str, Any], rounds: int = 50
                    ) -> float:
    """Calculates median of oof scores from multiple cross validations with random seeds."""
    scores = []

    model = model_class(params)
    for i in range(rounds):
        seed = choice(range(10000))

        _, _, score = model.score_cv(train_tokens, train_tags, seed=seed)

        scores.append(score[0])

    return np.median(scores)


def best_hyperopt_params(model_class: Any,
                        train_tokens: Sequence[Tokens], train_tags: Sequence[Tag],
                        hyperopt_trials: pd.DataFrame,
                        trials_tested: int = 10, n_rounds: int = 50
                        ) -> Dict[str, Any]:
    """
    Returns trial with most stable results.

    For each of top ``trials_tested`` hyperopt trials runs ``n_rounds`` cv, each with different seed.
    Returns params of trial with the highest median of oof score.

    Parameters:
        model_class (Any): classifier class
        train_tokens (Sequence[Tokens]): tokens to train on
        train_tags (Sequence[Tag]): labels to train on
        hyperopt_trials (pd.DataFrame): result of hyperopt search
        trials_tested (int): how many trials to test
        n_rounds (int): number of cv tested for each trial

    Returns:
        params (Dict[str, Any]): most stable params
    """

    trial_params = []
    trial_scores = []

    for i in range(trials_tested):
        params = ast.literal_eval(hyperopt_trials.loc[i, 'params'])
        trial_params.append(params)

        score = median_cv_score(model_class, train_tokens, train_tags, params, n_rounds)
        trial_scores.append(score)

    best_score = max(trial_scores)
    best_score_index = trial_scores.index(best_score)
    best_params = trial_params[best_score_index]

    return best_params



