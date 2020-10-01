import random

from glob import glob
from typing import List, Union, Callable
from collections import defaultdict

from mcqa_utils.answer import Answer
from mcqa_utils.utils import label_to_id, id_to_label
from mc_transformers.utils_mc import processors, DataProcessor, InputExample


# wrapper class around transfomers' DataProcessor
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

    def _make_contiguous_ids(self, global_id, examples):
        correct_examples = []
        # make ids contiguous
        for index, example in enumerate(examples):
            ex_id = self.processor._decode_id(example.example_id)
            context_id, question_id = int(ex_id[0]), int(ex_id[1])
            context_id += global_id
            correct_examples.append(
                InputExample(
                    example_id=self.processor._encode_id(
                        context_id, question_id
                    ),
                    question=example.question,
                    contexts=example.contexts,
                    endings=example.endings,
                    label=example.label,
                )
            )
            if index == (len(examples) - 1):
                global_id = context_id
        return global_id, correct_examples

    # these get functions are probably not needed
    def get_train_examples(self) -> List[InputExample]:
        return self.processor.get_train_examples(self.data_path)

    def get_dev_examples(self) -> List[InputExample]:
        return self.processor.get_dev_examples(self.data_path)

    def get_test_examples(self) -> List[InputExample]:
        return self.processor.get_test_examples(self.data_path)

    def get_all_examples(self, dir=None) -> List[InputExample]:
        if dir is None:
            dir = self.data_path + '/*.json'
        globbed_dir = dir
        glob_id = 0
        all_examples = []
        for file in glob(globbed_dir):
            examples = self.processor._read_examples(file, 'glob')
            print(f'Parsed {file}, found {len(examples)} examples')
            glob_id, examples = self._make_contiguous_ids(glob_id, examples)
            all_examples.extend(examples)
        return all_examples

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

    def get_gold_answers(
        self,
        splits: Union[List[str], str],
        with_text_values: bool = False
    ) -> List[Answer]:
        answers = []
        data = self.get_splits(splits)
        for example in data:
            example_id = '-'.join(self.decode_id(example.example_id))
            answer_dict = dict(
                example_id=example_id,
                label=example.label,
                pred_label=example.label
            )
            if with_text_values:
                answer_dict.update(endings=example.endings)

            answers.append(Answer(**answer_dict))

        return answers

    def find_mask(self, examples: List[InputExample], test_fn: Callable):
        mask = []
        for sample in examples:
            if test_fn(sample):
                mask.append(1)
            else:
                mask.append(0)
        return mask

    def to_json(self, examples):
        json_examples = {'version': 1.0, 'data': []}
        raw_examples = defaultdict(list)
        for sample in examples:
            context_id, _ = self.processor._decode_id(sample.example_id)
            try:
                context_id = int(context_id)
            except Exception:
                pass
            raw_examples[context_id].append(sample)
        for id, grouped in raw_examples.items():
            json_ex = {
                'id': id,
                'article': grouped[0].contexts[0],
                'answers': [id_to_label(ex.label) for ex in grouped],
                'options': [ex.endings for ex in grouped],
                'questions': [ex.question for ex in grouped],
            }
            json_examples['data'].append(json_ex)
        return json_examples

    def split_examples(self, examples, proportions, seed=None):
        if seed is not None:
            random.seed(seed)
        randomized = random.sample(examples, len(examples))
        if round(sum(proportions)) != 1:
            raise ValueError(
                'Proportions must sum to 1 for splitting! '
                f'Got ({proportions} = {sum(proportions)}) instead.'
            )
        splits = []
        start_idx, end_idx = 0, 0
        total_size = len(randomized)
        for prop in proportions:
            split_size = round(total_size * prop)
            end_idx = min(end_idx + split_size, total_size)
            splits.append(randomized[start_idx:end_idx])
            start_idx = end_idx

        return splits

    def iter_examples(self, examples):
        for ex in examples:
            num_labels = '0123456789'
            if ex.label in num_labels:
                label = int(ex.label)
            elif not isinstance(ex.label, int):
                label = ord(ex.label.upper()) - ord('A')
            yield ex.example_id, ex.contexts[0], ex.question, ex.endings, label

    def reduce_by_mask(
        self, data: List, mask: Union[List[bool], Callable]
    ) -> List:
        if isinstance(mask, Callable):
            mask = self.find_mask(data, mask)
        return self._reduce_by_mask_list(data, mask)

    def _reduce_by_mask_list(self, data: List, mask: List[bool]):
        end_list = []
        for point, keep in zip(data, mask):
            if bool(keep):
                end_list.append(point)

        return end_list

    # deprecated
    def apply_no_answer(
        self,
        split: Union[List[str], str],
        answers: List[Answer],
        text: str,
    ):
        data = self.get_splits(split)
        if len(data) != len(answers):
            raise ValueError(
                'Asked to set no answer on a list with different size '
                'from dataset, maybe you asked for the wrong split?'
                f'(dataset size {len(data)}, nof answers: {len(answers)})'
            )
        for datapoint, answer in zip(data, answers):
            assert(str(datapoint.example_id) == str(answer.example_id))
            ans_index = label_to_id(datapoint.label)
            answer_text = datapoint.endings[ans_index]
            found = answer_text.find(text) != -1
            if found and answer.get_answer() == ans_index:
                print(f'Aplying no answer to {answer.example_id}')
                answer.is_no_answer = True
        return answers
