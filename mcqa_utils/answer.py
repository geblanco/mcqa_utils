from typing import List, Tuple
from mcqa_utils.utils import argmax, label_to_id
from mc_transformers.utils_mc import InputExample


class Answer(object):
    def __init__(
        self,
        example_id: str,
        pred_label: str,
        label: str = None,
        probs: List[float] = None,
        threshold: float = 0.0,
        no_answer: float = -1.0,
        is_no_answer: bool = False,
    ):
        self.example_id = example_id
        self.probs = probs
        self.label = label
        self.pred_label = pred_label
        self.threshold = 0.0
        self.no_answer = -1.0
        self.is_no_answer = is_no_answer

    def get_answer(self) -> int:
        ans = label_to_id(self.pred_label)
        if self.is_no_answer:
            ans = self.no_answer
        elif self.probs is not None:
            ans = self.no_answer
            if max(self.probs) > self.threshold:
                ans = argmax(self.probs)
        return ans

    def get_pred_tuple(self) -> List[Tuple[str, float]]:
        return [(label_to_id(self.label), self.get_answer())]

    def get_example_id_answer_tuple(self) -> Tuple[str, str]:
        return (self.example_id, self.pred_label)

    def get_max_prob(self) -> float:
        return max(self.probs)

    @staticmethod
    def clone(answer):
        return Answer(
            example_id=answer.example_id,
            probs=answer.probs,
            label=answer.label,
            pred_label=answer.pred_label,
            threshold=answer.threshold,
            no_answer=answer.no_answer,
            is_no_answer=answer.is_no_answer,
        )


def parse_answer(answer_id, answer_value):
    answer_dict = dict(example_id=answer_id, pred_label=answer_value)
    if not isinstance(answer_value, (int, str)):
        answer_dict.update(
            probs=answer_value['probs'],
            pred_label=answer_value['pred_label'],
            label=answer_value['label']
        )
    return Answer(**answer_dict)


def input_example_to_answer(example: InputExample) -> Answer:
    return Answer(
        example_id=example.example_id,
        label=example.label,
        pred_label=example.label,
    )


def apply_threshold_to_answers(answers: List[Answer], threshold: float):
    for ans in answers:
        ans.threshold = threshold
    return answers
