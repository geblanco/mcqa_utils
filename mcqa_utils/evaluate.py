from typing import List, Union

from mcqa_utils.metric import Metric
from mcqa_utils.answer import Answer


class Evaluator(object):

    def __init__(self, metrics: Union[Metric, List[Metric]]):
        if isinstance(metrics, Metric):
            metrics = [metrics]
        self.metrics = metrics

    def evaluate(
        self,
        gold_answers: List[Answer],
        answers: List[Answer],
    ) -> float:
        raise NotImplementedError('You must implement `evaluate` method!')


class GenericEvaluator(Evaluator):

    def evaluate(
        self,
        gold_answers: List[Answer],
        answers: List[Answer],
    ) -> float:
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(gold_answers, answers)
        return results
