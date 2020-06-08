from tqdm import tqdm
from typing import List, Union

from mcqa_utils.dataset import Dataset
from mcqa_utils.question_answering import QASystem
from mcqa_utils.utils_multiple_choice import InputExample


class Evaluator(object):

    def __init__(self, metric):
        self.metric = metric

    def evaluate(
        self,
        qa_system: QASystem,
        dataset: Dataset,
        splits: Union[List[str], str] = None,
        mask: List[int] = None,
    ):
        raise NotImplementedError('You must implement `evaluate` method!')

    def apply_mask(self, data: List, mask: List) -> List:
        raise NotImplementedError()

    def reduce(self, data: List, mask: List) -> List:
        end_list = []
        for point, keep in zip(data, mask):
            if bool(keep):
                end_list.append(point)
        return end_list

    def forward_examples_for_answers(
        self,
        qa_system: QASystem,
        data: List[InputExample]
    ):
        # ToDo := parallelize
        answers = []
        for datapoint in tqdm(data, desc='Extracting answers'):
            answers.append(int(qa_system.get_answer(datapoint.example_id)))
        return answers


class GenericEvaluator(Evaluator):

    def evaluate(
        self,
        qa_system: QASystem,
        dataset: Dataset,
        splits: Union[List[str], str] = None,
        mask: List = None,
        keep: List = None,
    ):
        # ToDo := tqdm and parallel?
        # ToDo := apply mask
        results = []
        data = dataset.get_splits(splits)
        gold_answers = [int(ex.label) for ex in data]
        answers = self.forward_examples_for_answers(qa_system, data)
        if mask is not None:
            gold_answers = self.apply_mask(gold_answers, mask)
            answers = self.apply_mask(answers, mask)
        if keep is not None:
            gold_answers = self.reduce(gold_answers, keep)
            answers = self.reduce(answers, keep)
        return self.metric(gold_answers, answers)
