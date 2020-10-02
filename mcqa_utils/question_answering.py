import os
import json
import numpy as np

from collections import defaultdict
from typing import Union, Tuple, List
from mcqa_utils.utils import label_to_id, id_to_label
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
        self.missing_strategy = None
        if offline and answers_path is None:
            raise ValueError(
                'You must provide a path to the answers '
                'if the system is offline!'
            )

    def get_answer(self, example_id: Union[str, int]) -> Answer:
        raise NotImplementedError(
            'You must implement `get_answer` method!')

    def fill_missing(self, example_id: Union[str, int]):
        raise NotImplementedError(
            'You must implement `fill_missing` method!')

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
            answer = None
            if self.missing_strategy is not None:
                answer = self.fill_missing(example_id, self.get_nof_choices())
            if answer is not None:
                self.answers[example_id] = answer
            else:
                raise ValueError('Example not found %r' % example_id)

        return self.answers[example_id]

    def get_nof_choices(self) -> int:
        first_answer = self.answers[list(self.answers.keys())[0]]
        if first_answer.probs is not None:
            max_value = len(first_answer.probs)
        else:
            max_value = -1
            for ans in self.answers.values():
                ans_value = label_to_id(ans.pred_label)
                if ans_value > max_value:
                    max_value = ans_value
        return max_value

    def fill_missing(
        self, example_id: Union[str, int], nof_choices: int
    ) -> Answer:
        answer_dict = dict(
            example_id=example_id,
        )
        if self.missing_strategy.lower() == 'uniform':
            probs = np.random.uniform(low=0, high=1.0, size=(nof_choices))
            probs /= sum(probs)
        else:
            if self.missing_strategy.lower() == 'random':
                value = np.random.randint(nof_choices)
                probs = np.zeros(nof_choices)
                probs[value] = 1.0
            else:
                value = int(self.missing_strategy)
                probs = np.array([value] * nof_choices)
                if sum(probs) > 0:
                    probs /= sum(probs)

        answer_dict.update(
            probs=probs.tolist(),
            pred_label=id_to_label(np.argmax(probs))
        )

        return Answer(**answer_dict)

    def parse_predictions(self, raw_answers: dict) -> dict:
        answers = {}
        first_answer_key = list(raw_answers.keys())[0]
        # raw_answers can come as predictions or nbest-predictions
        # when working with reduced datasets, extra null values
        # are kept to properly match answer ids: if ids are
        # composed based on the position of the answer (as done
        # with enumerate in parse method and in transformers' processor,
        # ids will be unaligned, keeping nulls to avoids this, marking
        # it as a skip index
        if isinstance(raw_answers[first_answer_key], list):
            for context_id, context_answers in raw_answers.items():
                for answer_id, answer_value in enumerate(context_answers):
                    if answer_value is None:
                        continue
                    if answer_id < 10:
                        answer_id = f'0{answer_id}'
                    qas_id = f'{context_id}-{answer_id}'
                    answers[qas_id] = parse_answer(qas_id, answer_value)
        else:
            for ans_id, ans_value in raw_answers.items():
                if ans_value is None:
                    continue
                answers[ans_id] = parse_answer(ans_id, ans_value)

        return answers

    def unparse_predictions(self, answers: list = None) -> dict:
        if answers is None:
            answers = self.get_all_answers()
        first_answer = answers[0]
        if isinstance(unparse_answer(first_answer), (int, str)):
            output_dict = {}
            is_nbest_predictions = False
        else:
            output_dict = defaultdict(list)
            is_nbest_predictions = True
        # ids come as <ans_id>-<ex_id> because a context has multiple answers
        for answer_id, ans in enumerate(answers):
            ans_id = str(ans.example_id)
            if is_nbest_predictions:
                ans_id = str(ans_id).split('-')[0]
                output_dict[ans_id].append(unparse_answer(ans))
            else:
                output_dict[ans_id] = unparse_answer(ans)

        return output_dict

    def unparse_predictions_with_alignment(
        self, gold_answers: list, answers: list = None
    ) -> dict:
        if answers is None:
            answers = self.get_all_answers()
        first_answer = answers[0]
        if isinstance(unparse_answer(first_answer), (int, str)):
            output_dict = {}
            is_nbest_predictions = False
        else:
            output_dict = defaultdict(list)
            is_nbest_predictions = True
        # ids come as <ans_id>-<ex_id> because a context has multiple answers
        # traverse gold answers searching for valid answers and filling nulls
        answer_index = 0
        for gold in gold_answers:
            gold_id = str(gold.example_id)
            if (
                answer_index >= len(answers) or
                str(answers[answer_index].example_id) != gold_id
            ):
                to_append = None
            else:
                to_append = unparse_answer(answers[answer_index])
                answer_index += 1

            if is_nbest_predictions:
                gold_id = str(gold_id).split('-')[0]
                output_dict[gold_id].append(to_append)
            else:
                output_dict[gold_id] = to_append

        return output_dict

    def load_predictions(self, path: str) -> dict:
        full_path = os.path.abspath(path)
        with open(full_path, 'r') as fstream:
            answers = json.load(fstream)
        return answers
