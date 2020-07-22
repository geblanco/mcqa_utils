from typing import List, Tuple
from mcqa_utils.utils import argmax, label_to_id


def parse_answer(answer_id, answer_value):
    answer_dict = dict(example_id=answer_id, pred_label=answer_value)
    if not isinstance(answer_value, (int, str)):
        answer_dict.update(
            probs=answer_value['probs'],
            pred_label=answer_value['pred_label'],
            label=answer_value['label']
        )
    return Answer(**answer_dict)


class Answer(object):
    def __init__(
        self,
        example_id: str,
        pred_label: str,
        label: str = None,
        probs: List[float] = None,
        no_answer: float = -1.0
    ):
        self.example_id = example_id
        self.probs = probs
        self.label = label
        self.pred_label = pred_label
        self.threshold = 0.0
        self.no_answer = -1.0

    def get_answer(self) -> int:
        ans = label_to_id(self.pred_label)
        if self.probs is not None:
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
