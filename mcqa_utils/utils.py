import json

from typing import List, Union
from collections import defaultdict
from mcqa_utils.answer import Answer


def argmax(values: List[Union[float, int]]) -> Union[float, int]:
    max_val, max_idx = values[0], 0
    for i in range(len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_idx = i
    return max_idx


def label_to_id(label: str) -> int:
    return ord(label.upper()) - ord('A')


def unique(values):
    ret = {}
    for value in values:
        ret[value] = 1
    return list(ret.keys())


def flatten_dict(lists, keys=None):
    flat_array = []
    if keys is None:
        flat_array = [item for _list in lists for item in lists[_list]]
    else:
        for key in keys:
            flat_array.extend(lists[key])
    return flat_array


def sort_dict(_dict):
    ret = {}
    for key in sorted(_dict.keys()):
        ret[key] = _dict[key]
    return ret


def parse_predictions_file(predictions_file):
    predictions = json.load(open(predictions_file, 'r'))
    predictions = [
        Answer(id=p['id'], probs=p['probs'],
               label=p['label'], pred_label=p['pred_label'])
        for p in predictions
    ]
    return predictions


def get_labels_from_dataset(dataset):
    id_ans = {}
    for test in dataset:
        id_ans[str(test['id'])] = [label_to_id(a) for a in test['answers']]
    return id_ans


def get_labels_from_answers(answers):
    id_ans = defaultdict(list)
    for ans in answers:
        id_ans[str(ans.id)].append(ans.get_answer())
    return id_ans


def get_labels(dataset_or_answers):
    if type(dataset_or_answers[0]) == Answer:
        return get_labels_from_answers(dataset_or_answers)
    else:
        return get_labels_from_dataset(dataset_or_answers)
