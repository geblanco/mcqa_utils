import numpy as np

from typing import List, Union
from functools import partial
from mc_transformers.utils_mc import InputExample


def argmax(values: List[Union[float, int]]) -> Union[float, int]:
    max_val, max_idx = values[0], 0
    for i in range(len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_idx = i
    return max_idx


def label_to_id(label: Union[str, int, float]) -> int:
    if isinstance(label, (int, np.integer, float, np.floating)):
        return int(label)
    else:
        try:
            # numeric value comes as string
            label = int(label)
        except ValueError:
            # label is in ['A', 'B', 'C', 'D'...]
            label = ord(label.upper()) - ord('A')
        finally:
            return label


def id_to_label(label: int) -> str:
    if isinstance(label, str):
        try:
            # numeric value comes as string
            label = int(label)
        except ValueError:
            # it was in ['A', 'B', 'C', 'D'...]
            pass
    if isinstance(label, (float, np.floating)):
        label = int(label)
    if isinstance(label, (int, np.integer)):
        # it was originally a number or a number as string
        # converted to int
        label = chr(ord('A') + label)
    return label


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


def flatten(lists):
    return [elem for sublist in lists for elem in sublist]


def sort_dict(_dict):
    ret = {}
    for key in sorted(_dict.keys()):
        ret[key] = _dict[key]
    return ret


def answer_mask_fn(mask_cfg, sample):
    mask_text = mask_cfg['text'].lower()
    keep_if_found = mask_cfg['match']
    ans_index = label_to_id(sample.label)
    answer = sample.endings[ans_index].lower()
    found = answer.find(mask_text) != -1
    keep = (found and keep_if_found) or (not found and not keep_if_found)
    return keep


def get_mask_matching_text(answer_text: str, match: bool):
    return partial(
        answer_mask_fn, {'text': answer_text, 'match': match}
    )


def update_example(example, **kwargs):
    dict_example = example.todict()
    dict_example.update(**kwargs)
    return InputExample(
        example_id=dict_example['example_id'],
        question=dict_example['question'],
        contexts=dict_example['contexts'],
        endings=dict_example['endings'],
        label=dict_example['label'],
    )
