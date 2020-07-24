
from typing import List
from mcqa_utils.answer import Answer


class Metric(object):

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        raise NotImplementedError()

    def needs_no_answer(self):
        return False


class Metric_with_no_answer(Metric):

    no_answer = -1

    def needs_no_answer(self):
        return True


class C_at_1(Metric_with_no_answer):

    name = 'C_at_1'

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        correct = 0
        unanswered = 0
        total = len(gold_answers)
        for gold_ans, ans in zip(gold_answers, answers):
            answer_value = ans.get_answer()
            if gold_ans.get_answer() == answer_value:
                correct += 1
            elif answer_value == self.no_answer:
                unanswered += 1
        return (1 / total) * (correct + (correct / total) * unanswered)


class Average(Metric):

    name = 'avg'

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        correct = 0
        total = len(gold_answers)
        for gold_ans, ans in zip(gold_answers, answers):
            if gold_ans.get_answer() == ans.get_answer():
                correct += 1
        return correct / total


class F1(Metric_with_no_answer):

    name = 'f1'

    def __call__(self, gold_answers: List[Answer], answers: List[Answer]):
        if self.no_answer is None:
            raise ValueError(
                'To calculate F1 score you need `no_answer` '
                f'(given = {self.no_answer}'
            )
        # true_pos = 0
        # true_answers = [answer for answer in gold_answers
        #                     if answer.get_answer() == self.no_answer]
        # for gold_ans in true_answers:
        #     pass


metrics_map = {
    'C_at_1': C_at_1,
    'f1': F1,
    'avg': Average,
}
