from typing import List, Union

from mcqa_utils.metric import Metric
from mcqa_utils.dataset import Dataset
from mcqa_utils.question_answering import QASystem
from mc_transformers.utils_mc import InputExample


class Evaluator(object):

    def __init__(self, metric: Metric):
        self.metric = metric

    def evaluate(
        self,
        qa_system: QASystem,
        dataset: Dataset,
        splits: Union[List[str], str] = None,
        mask: List[int] = None,
    ) -> float:
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
    ) -> List[int]:
        # ToDo := parallelize
        answers = []
        not_found = []
        for datapoint in data:
            try:
                answer = qa_system.get_answer(datapoint.example_id)
                answers.append(answer)
            except ValueError:
                not_found.append(datapoint.example_id)
            except NotImplementedError as ex:
                raise ex
        return answers, not_found


class GenericEvaluator(Evaluator):

    def evaluate(
        self,
        qa_system: QASystem,
        dataset: Dataset,
        splits: Union[List[str], str] = None,
        mask: List = None,
        keep: List = None,
        quiet: bool = False,
    ) -> float:
        # ToDo := parallel?
        # ToDo := apply mask
        # Difference between mask and keep:
        # - mask doesn't reduce dimensionality
        # - keep will drop masked answers, so dimensionality will be reduced
        data = dataset.get_splits(splits)
        gold_answers = [int(ex.label) for ex in data]
        answers, not_found = self.forward_examples_for_answers(qa_system, data)
        if mask is not None:
            gold_answers = self.apply_mask(gold_answers, mask)
            answers = self.apply_mask(answers, mask)
        n_answers = len(answers)
        if keep is not None:
            gold_answers = self.reduce(gold_answers, keep)
            answers = self.reduce(answers, keep)
            if not quiet:
                print(f'Reduced by mask from {n_answers} to {len(answers)}')
                print(f'Number of answers not found {len(not_found)}')
        return self.metric(gold_answers, answers)
