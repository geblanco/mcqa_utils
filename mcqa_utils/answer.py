from typing import List, Tuple, Optional, Union
from mcqa_utils.utils import argmax, label_to_id, id_to_label


class Answer(object):
    def __init__(
        self,
        example_id: str,
        pred_label: str,
        label: str = None,
        probs: List[float] = None,
        endings: Optional[List[str]] = None,
        logits: Optional[List[float]] = None,
        threshold: float = 0.0,
        no_answer: float = -1.0,
        no_answer_text: str = None,
        is_no_answer: bool = False,
    ):
        self.example_id = example_id
        self.pred_label = pred_label
        self.label = label
        self.probs = probs
        self.endings = endings
        self.logits = logits
        self.threshold = threshold
        self.no_answer = no_answer
        self.no_answer_text = no_answer_text
        self.probs_field = 'probs'
        self.is_no_answer = is_no_answer
        if not is_no_answer and self.pred_label is not None:
            self.is_no_answer = label_to_id(self.pred_label) == self.no_answer

    def get_answer(self, accept_no_answer=True) -> int:
        ans = label_to_id(self.pred_label)
        if self.is_no_answer:
            ans = self.no_answer
        elif self.__getattribute__(self.probs_field) is not None:
            ans = self.no_answer
            if max(self.__getattribute__(self.probs_field)) > self.threshold:
                ans = argmax(self.__getattribute__(self.probs_field))

        if ans == self.no_answer and not accept_no_answer:
            ans = self.search_unanswerable_option()
        return ans

    def get_pred_tuple(self) -> List[Tuple[str, float]]:
        return [(label_to_id(self.label), self.get_answer())]

    def get_example_id_answer_tuple(self) -> Tuple[str, str]:
        return (self.example_id, self.pred_label)

    def get_max_prob(self) -> float:
        return max(self.__getattribute__(self.probs_field))

    def get_min_prob(self) -> float:
        return min(self.__getattribute__(self.probs_field))

    def search_unanswerable_option(self):
        unanswerable_option_index = self.no_answer
        if self.no_answer_text is not None and self.endings is not None:
            for idx, end in enumerate(self.endings):
                if end.lower() == self.no_answer_text.lower():
                    unanswerable_option_index = idx
                    break

        return unanswerable_option_index

    @staticmethod
    def clone(answer):
        probs = answer.probs.copy() if answer.probs is not None else None
        ends = answer.endings.copy() if answer.endings is not None else None
        logits = answer.logits.copy() if answer.logits is not None else None
        return Answer(
            example_id=answer.example_id,
            probs=probs,
            endings=ends,
            logits=logits,
            label=answer.label,
            pred_label=answer.pred_label,
            threshold=answer.threshold,
            no_answer=answer.no_answer,
            no_answer_text=answer.no_answer_text,
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
        if 'logits' in answer_value:
            answer_dict.update(logits=answer_value['logits'])
    return Answer(**answer_dict)


def unparse_answer(answer: Answer) -> Union[dict, int, str]:
    output = answer.pred_label
    if answer.probs is not None:
        output = dict(pred_label=answer.pred_label, probs=answer.probs)
        if answer.label is not None:
            output.update(label=id_to_label(answer.label))
        if answer.logits is not None:
            output.update(logits=answer.logits)
    return output


def apply_threshold_to_answers(answers: List[Answer], threshold: float):
    for ans in answers:
        ans.threshold = threshold
    return answers


def apply_no_answer(answers: List[Answer], no_answer_label_ids: List[str]):
    for ans, no_ans_label_id in zip(answers, no_answer_label_ids):
        if ans.get_answer() == no_ans_label_id:
            ans.is_no_answer = True


def apply_prob_field_to_answers(answers: List[Answer], field: str):
    for ans in answers:
        ans.probs_field = field
