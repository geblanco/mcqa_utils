from typing import List, Union


def argmax(values: List[Union[float, int]]) -> Union[float, int]:
    max_val, max_idx = values[0], 0
    for i in range(len(values)):
        if values[i] > max_val:
            max_val = values[i]
            max_idx = i
    return max_idx


def label_to_id(label: Union[str, int, float]) -> int:
    if isinstance(label, (int, float)):
        return label
    else:
        try:
            # numeric value comes as string
            label = int(label)
        except ValueError:
            # label is in ['A', 'B', 'C', 'D'...]
            label = ord(label.upper()) - ord('A')
        finally:
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
