import os
import json

from collections import defaultdict
from typing import Union, Tuple, List
from mcqa_utils.answer import (
    parse_answer,
    unparse_answer,
    Answer,
)


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
        raise NotImplementedError(
            'You must implement `get_answer` method!')

    def get_answers(
        self,
        data: List[Answer],
        with_text_values: bool = False,
        no_answer_text: str = None
    ) -> Tuple[List[Answer], List[Union[str, int]]]:
        # ToDo := parallelize
        answers = []
        not_found = []
        if with_text_values and (
            data[0].endings is None or
            no_answer_text is None
        ):
            raise ValueError(
                'Asked for answers with text options but dataset doesn\'t '
                'contain text endings or `no_answer_text` was not provided,'
                ' pass `dataset.get_gold_answers(with_text_values=True) to '
                'parse text endings in dataset!'
            )
        for datapoint in data:
            try:
                answer = self.get_answer(datapoint.example_id)
                if with_text_values:
                    answer.endings = datapoint.endings
                    answer.no_answer_text = no_answer_text
                answers.append(answer)
            except ValueError:
                not_found.append(datapoint.example_id)
            except NotImplementedError as ex:
                raise ex
        return answers, not_found

    def get_all_answers(self) -> List[Answer]:
        return list(self.answers.values())


class QASystemForMCOffline(QASystem):

    def __init__(self, answers_path: str):
        offline = True
        super(QASystemForMCOffline, self).__init__(offline, answers_path)
        raw_answers = self.load_predictions(self.answers_path)
        self.answers = self.parse_predictions(raw_answers)

    def get_answer(self, example_id: Union[str, int]) -> Answer:
        # answers are in a dict, ensure string index access
        example_id = str(example_id)
        if example_id not in self.answers:
            raise ValueError('Example not found %r' % example_id)
        else:
            return self.answers[example_id]

    def parse_predictions(self, raw_answers: dict) -> dict:
        answers = {}
        first_answer_key = list(raw_answers.keys())[0]
        # raw_answers can come as predictions or nbest-predictions
        if isinstance(raw_answers[first_answer_key], list):
            for context_id, context_answers in raw_answers.items():
                for answer_id, answer_value in enumerate(context_answers):
                    if answer_id < 10:
                        answer_id = f'0{answer_id}'
                    qas_id = f'{context_id}-{answer_id}'
                    answers[qas_id] = parse_answer(qas_id, answer_value)
        else:
            for ans_id, ans_value in raw_answers.items():
                answers[ans_id] = parse_answer(ans_id, ans_value)

        return answers

    def unparse_predictions(self, answers: list = None) -> dict:
        if answers is None:
            answers = list(self.answers.values())
        first_answer = answers[0]
        if isinstance(unparse_answer(first_answer), (int, str)):
            output_dict = {}
            is_nbest_predictions = False
        else:
            output_dict = defaultdict(list)
            is_nbest_predictions = True
        for ans in answers:
            ans_id = str(ans.example_id)
            if is_nbest_predictions:
                ans_id = str(ans_id).split('-')[0]
                output_dict[ans_id].append(unparse_answer(ans))
            else:
                output_dict[ans_id] = unparse_answer(ans)
        return output_dict

    def load_predictions(self, path: str) -> dict:
        full_path = os.path.abspath(path)
        with open(full_path, 'r') as fstream:
            answers = json.load(fstream)
        return answers
