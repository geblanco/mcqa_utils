from typing import List, Tuple


# ToDo := Unify label and pred_label to int or string field
class Answer(object):
    def __init__(
        self,
        example_id: str,
        probs: List[float],
        label: str,
        pred_label: str,
        no_answer: float = -1.0
    ):
        self.example_id = example_id
        self.probs = probs
        self.label = label
        self.pred_label = pred_label
        self.threshold = 0.0
        self.no_answer = -1.0

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def get_answer(self) -> float:
        from mcqa_utils.utils import argmax
        ans = self.no_answer
        if max(self.probs) > self.threshold:
            ans = argmax(self.probs)
        return ans

    def get_pred_tuple(self) -> List[Tuple[str, float]]:
        return [(self.label, self.get_answer())]

    def get_example_id_answer_tuple(self) -> Tuple[str, str]:
        return (self.example_id, self.pred_label)

    def get_max_prob(self) -> float:
        return max(self.probs)
