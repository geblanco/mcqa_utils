import concurrent.futures as cf

from functools import partial

from typing import List, Tuple

from mcqa_utils.metric import Metric
from mcqa_utils.evaluate import Evaluator
from mcqa_utils.utils import argmax, unique, flatten
from mcqa_utils.answer import Answer, apply_threshold_to_answers


def sweeper(metric, gold_answers, answers, increments):
    scores = []
    clones = [Answer.clone(ans) for ans in answers]
    for threshold in increments:
        for ans in clones:
            ans.threshold = threshold
        scores.append(metric(gold_answers, clones))
    return scores


class Threshold(object):

    def __init__(self, evaluator: Evaluator, nof_threads=8):
        self.evaluator = evaluator
        self.nof_threads = nof_threads

    def _concurrent_sweep(
        self,
        metric: Metric,
        gold_answers: List[Answer],
        answers: List[Answer],
        increments: List[float]
    ) -> Tuple[int, List[float]]:
        with cf.ProcessPoolExecutor(max_workers=self.nof_threads) as executor:
            partial_sweep = partial(sweeper, metric, gold_answers, answers)
            batch_size = len(increments) // self.nof_threads
            # divide to increments in batches to maximize i/o
            increment_steps = [
                increments[i:min(i + batch_size, len(increments))]
                for i in range(0, len(increments), batch_size)
            ]
            scores = flatten(executor.map(partial_sweep, increment_steps))
            best_thresh_idx = argmax(scores)
        return best_thresh_idx, scores

    def _sweep(
        self,
        metric: Metric,
        gold_answers: List[Answer],
        answers: List[Answer],
        increments: List[float]
    ) -> Tuple[int, List[float]]:
        scores = []
        for threshold in increments:
            apply_threshold_to_answers(answers, threshold)
            scores.append(metric(gold_answers, answers))
        best_thresh_idx = argmax(scores)
        return best_thresh_idx, scores

    def find_best_threshold(
        self,
        metric: Metric,
        gold_answers: List[Answer],
        answers: List[Answer],
    ) -> float:
        max_probs = [ans.get_max_prob() for ans in answers]
        increments = unique([0] + sorted(max_probs))
        sweep_function = self._sweep
        if len(answers) > 5000:
            sweep_function = self._concurrent_sweep

        best_thresh_idx, scores = sweep_function(
            metric, gold_answers, answers, increments
        )
        # reset to previous thresholds,
        # unnecessary since all answers are cloned
        apply_threshold_to_answers(answers, 0.0)
        return increments[best_thresh_idx]
