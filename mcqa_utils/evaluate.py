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
        keep: List = None,
        quiet: bool = True,
    ) -> float:
        raise NotImplementedError('You must implement `evaluate` method!')

    def reduce(self, data: List, mask: List) -> List:
        end_list = []
        for point, keep in zip(data, mask):
            if bool(keep):
                end_list.append(point)
        return end_list


class GenericEvaluator(Evaluator):

    def evaluate(
        self,
        gold_answers: List[Answer],
        answers: List[Answer],
        keep: List = None,
        quiet: bool = True,
    ) -> float:
        # - keep will drop masked answers, so dimensionality will be reduced
        n_answers = len(answers)
        if keep is not None:
            gold_answers = self.reduce(gold_answers, keep)
            answers = self.reduce(answers, keep)
            if not quiet:
                print(f'Reduced by mask from {n_answers} to {len(answers)}')
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric(gold_answers, answers)
        return results
