from typing import List, Union, Callable
from mcqa_utils.utils import label_to_id
from mcqa_utils.metric import Metric
from mcqa_utils.question_answering import QASystem
from mcqa_utils.utils_multiple_choice import processors, DataProcessor, InputExample


class Dataset(object):

    def __init__(
        self,
        data_path: str,
        task: str,
        processor: DataProcessor = None,
        name: str = None
    ):
        self.data_path = data_path
        self.task = task
        if processor is None:
            self.processor_cls = processors[task]
        else:
            self.processor_cls = processor
        self.processor = self.processor_cls()
        self.name = task if name is None else name
        self.splits = ['train', 'dev', 'test']
        self.split_to_get_map = {
            'train': self.get_train_examples,
            'dev': self.get_dev_examples,
            'test': self.get_test_examples,
        }

    # these get functions are probably not needed
    def get_train_examples(self) -> List[InputExample]:
        return self.processor.get_train_examples(self.data_path)

    def get_dev_examples(self) -> List[InputExample]:
        return self.processor.get_dev_examples(self.data_path)

    def get_test_examples(self) -> List[InputExample]:
        return self.processor.get_test_examples(self.data_path)

    def get_split(self, split: str) -> List[InputExample]:
        if split not in self.splits:
            raise ValueError('Unknown split! %r' % split)
        return self.split_to_get_map[split]()

    def get_splits(self, splits: List[str]) -> List[InputExample]:
        data = []
        if splits is None:
            splits = self.splits[1:]

        if not isinstance(splits, list):
            splits = [splits]

        for split in splits:
            data.extend(self.get_split(split))
        return data

    def get_available_labels(self) -> List[str]:
        return self.processor.get_labels()

    def get_labels(self, splits: Union[List[str], str] = None) -> List[int]:
        if isinstance(splits, str):
            splits = [splits]
        id_ans = {}
        for test in self.get_splits(splits):
            id_ans[test.example_id] = [label_to_id(a) for a in test]
        return id_ans

    def encode_id(self, id: str):
        return self.processor._encode_id(id)

    def decode_id(self, id: str) -> str:
        return self.processor._decode_id(id)

    def find_mask(self, split: Union[List[str], str], test_fn: Callable):
        mask = []
        for sample in self.get_splits(split):
            if test_fn(sample):
                mask.append(1)
            else:
                mask.append(0)
        return mask
