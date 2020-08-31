import os
import json
from typing import Union, Tuple, List
from mcqa_utils.answer import parse_answer, Answer

from mc_transformers.utils_mc import InputExample


class QASystem(object):
    def __init__(self, offline: bool = True, answers_path: str = None):
        self.answers = {}
        self.offline = offline
        self.answers_path = answers_path
        if offline and answers_path is None:
            raise ValueError(
                'You must provide a path to the answers '
                'if the system is offline!'
            )

    def get_answer(self, example_id: Union[str, int]) -> Answer:
        if self.offline:
            # answers are in a dict, ensure string index access
            example_id = str(example_id)
            if example_id not in self.answers:
                raise ValueError('Example not found %r' % example_id)
            else:
                return self.answers[example_id]
        else:
            raise NotImplementedError(
                'You must implement `get_answer` method!')

    def get_answers(
        self,
        data: List[InputExample],
    ) -> Tuple[List[Answer], List[Union[str, int]]]:
        # ToDo := parallelize
        answers = []
        not_found = []
        for datapoint in data:
            try:
                answer = self.get_answer(datapoint.example_id)
                answers.append(answer)
            except ValueError:
                not_found.append(datapoint.example_id)
            except NotImplementedError as ex:
                raise ex
        return answers, not_found


class QASystemForMCOffline(QASystem):

    def __init__(self, answers_path: str):
        offline = True
        super(QASystemForMCOffline, self).__init__(offline, answers_path)
        raw_answers = self.load_predictions(self.answers_path)
        self.answers = self.parse_predictions(raw_answers)

    def parse_predictions(raw_answers: dict) -> dict:
        answers = {}
        # raw_answers can come as predictions or nbest-predictions
        if isinstance(raw_answers[0], list):
            for context_id, context_answers in raw_answers:
                for answer_id, answer_value in enumerate(context_answers):
                    if answer_id < 10:
                        answer_id = f'0{answer_id}'
                    qas_id = f'{context_id}-{answer_id}'
                    answers[qas_id] = parse_answer(qas_id, answer_value)
        else:
            for ans_id, ans_value in raw_answers.items():
                answers[ans_id] = parse_answer(ans_id, ans_value)

        return answers

    def load_predictions(self, path: str) -> dict:
        full_path = os.path.abspath(path)
        with open(full_path, 'r') as fstream:
            answers = json.load(fstream)
        return answers
