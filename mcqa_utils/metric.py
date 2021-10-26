import numpy as np

from typing import List, Optional
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
from mcqa_utils.answer import Answer


@dataclass(frozen=True)
class MetricOutput:
    value: float
    total: Optional[int] = None
    correct: Optional[int] = None
    incorrect: Optional[int] = None
    unanswered: Optional[int] = None
    unanswered_correct: Optional[int] = None
    unanswered_incorrect: Optional[int] = None
    true_positive: Optional[List[int]] = None
    false_negative: Optional[List[int]] = None
    false_positive: Optional[List[int]] = None
    true_negative: Optional[List[int]] = None


class Metric_with_extras(object):
    has_extras = True

    def add_extras(self, results):
        """
        Careful, this method will receive all the previous results,
        can alter them
        """
        raise NotImplementedError()


class Metric(object):
    no_answer = None
    has_extras = False

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        raise NotImplementedError()

    def needs_no_answer(self):
        return self.no_answer is not None


class Metric_with_no_answer(Metric):
    no_answer = -1


class C_at_1(Metric_with_no_answer, Metric_with_extras):
    name = "C_at_1"
    has_extras = True

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        correct = 0
        unanswered = 0
        total = len(gold_answers)
        for gold_ans, ans in zip(gold_answers, answers):
            gold_value = gold_ans.get_answer()
            answer_value = ans.get_answer()
            if gold_value == answer_value:
                correct += 1
            elif answer_value == self.no_answer:
                unanswered += 1

        value = (1 / total) * (correct + (correct / total) * unanswered)
        incorrect = total - correct - unanswered

        return MetricOutput(
            value=value,
            total=total,
            correct=correct,
            incorrect=incorrect,
            unanswered=unanswered,
        )

    def add_extras(self, results):
        must_fields = [
            self.name + "_correct",
            self.name + "_incorrect",
            Average.name + "_correct",
            Average.name + "_incorrect",
        ]
        test = [results.get(field, None) is not None for field in must_fields]
        if all(test):
            un_correct = results[must_fields[2]] - results[must_fields[0]]
            un_incorrect = results[must_fields[3]] - results[must_fields[1]]
            results[self.name + "_unanswered_correct"] = un_correct
            results[self.name + "_unanswered_incorrect"] = un_incorrect

        return results


class UtilityFunction(Metric_with_no_answer):

    name = "utility_function"
    # unanswered, wrong, right
    utility = [0, -0.25, 1]

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        # save name to avoid collisions
        utility_str = '_'.join([str(u) for u in self.utility])
        self.name = f"utility_function_{utility_str}"
        correct = 0
        unanswered = 0
        total = len(gold_answers)
        for gold_ans, ans in zip(gold_answers, answers):
            gold_value = gold_ans.get_answer()
            answer_value = ans.get_answer()
            if gold_value == answer_value:
                correct += 1
            elif answer_value == self.no_answer:
                unanswered += 1
        incorrect = total - correct - unanswered
        iter_val = zip(self.utility, [unanswered, incorrect, correct])
        value = sum([ut * val for ut, val in iter_val]) / total

        return MetricOutput(
            value=value,
            total=total,
            correct=correct,
            incorrect=incorrect,
            unanswered=unanswered,
        )


class Average(Metric):

    name = "avg"

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        correct = 0
        total = len(gold_answers)
        for gold_ans, ans in zip(gold_answers, answers):
            # allow answers to search for the option with unanswerable text
            ans_opt = ans.get_answer(accept_no_answer=self.needs_no_answer())
            if gold_ans.get_answer() == ans_opt:
                correct += 1
        value = (correct / total)
        incorrect = total - correct
        return MetricOutput(
            value=value,
            total=total,
            correct=correct,
            incorrect=incorrect,
        )


class F1(Metric_with_no_answer):

    name = "f1"

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        if self.no_answer is None:
            raise ValueError(
                "To calculate F1 score you need `no_answer` "
                f"(given = {self.no_answer}"
            )
        # true_pos = 0
        # true_answers = [answer for answer in gold_answers
        #                     if answer.get_answer() == self.no_answer]
        # for gold_ans in true_answers:
        #     pass


class ConfusionMatrix(Metric):

    name = "confusion matrix"

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        true_labels = [
            gold.get_answer(accept_no_answer=self.needs_no_answer())
            for gold in gold_answers
        ]
        pred_labels = [
            ans.get_answer(accept_no_answer=self.needs_no_answer())
            for ans in answers
        ]
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels)
        if isinstance(tn, np.ndarray):
            tn = tn.tolist()
            fp = fp.tolist()
            fn = fn.tolist()
            tp = tp.tolist()

        return MetricOutput(
            value=0.0,
            true_positive=tp,
            false_negative=fn,
            false_positive=fp,
            true_negative=tn,
        )


metrics_map = {
    "C_at_1": C_at_1,
    "f1": F1,
    "avg": Average,
    "utility_function": UtilityFunction,
    "confusion_matrix": ConfusionMatrix,
}
