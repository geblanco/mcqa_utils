# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """


import csv
import glob
import json
import logging
import os
from typing import List

import tqdm
import re


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""
    def __init__(self):
        super(RaceProcessor, self).__init__()
        self.reg = re.compile(r'^.*/RACE/(.*)\.txt$')
        self.replacements = [
            ('train', '01'), ('dev', '02'), ('test', '03'),
            ('high', '04'), ('middle', '05'),
            ('/', '00')]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        data = []
        for path in self._get_paths_from_data_dir("train", data_dir):
            data.extend(self._read_txt(path))
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        data = []
        for path in self._get_paths_from_data_dir("dev", data_dir):
            data.extend(self._read_txt(path))
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        data = []
        for path in self._get_paths_from_data_dir("test", data_dir):
            data.extend(self._read_txt(path))
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _encode_race_id(self, race_id):
        """
          train/dev/test := 01, 02, 03
          high/middle    := 04, 02
          <int>.txt      := int
          /              := 00
        """
        race_id = self.reg.findall(race_id)[0]
        for repl in self.replacements:
            race_id = race_id.replace(repl[0], repl[1])
        return int(race_id)

    def _decode_race_id(self, race_id):
        race_id = '0' + str(race_id)
        id = race_id[8:] + '.txt'
        race_id = race_id[:8]
        for repl in self.replacements:
            race_id = race_id.replace(repl[1], repl[0])
        return race_id + id

    def _get_paths_from_data_dir(self, prefix, data_dir):
        paths = []
        if prefix not in data_dir:
            paths.append(os.path.join(data_dir, prefix, 'high'))
            paths.append(os.path.join(data_dir, prefix, 'middle'))
        else:
            # e.g.: data_dir = RACE/test/high
            paths = [data_dir]
        return paths

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = self._encode_race_id(data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


# data loader for:
# https://www.sciencedirect.com/science/article/abs/pii/S0306457317305344
# ONLY test examples available!, not really necessary, use GenericProcessor
class EntranceExamsProcessor(DataProcessor):
    """Processor for the Entrance Exams data set."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(data_dir))
        data = self._read_json(data_dir)
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            data_raw = json.load(fin)
        return data_raw

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for data_raw in data['data']:
            article = data_raw['article']
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=data_raw['id'],
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class GenericProcessor(DataProcessor):

    def _read_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(data_dir))
        data = self._read_json(data_dir)
        return self._create_examples(data)

    """Processor for the Generic data sets."""
    def get_train_examples(self, data_dir):
        file = os.path.join(data_dir, "train.json")
        return self._read_examples(file)

    def get_dev_examples(self, data_dir):
        file = os.path.join(data_dir, "dev.json")
        return self._read_examples(file)

    def get_test_examples(self, data_dir):
        file = os.path.join(data_dir, "test.json")
        return self._read_examples(file)

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _encode_generic_id(self, str_id):
        if not isinstance(str_id, str):
            str_id = str(str_id)
        id = ''
        # ascii representation of an alphabet char is always a number
        # of two/three ciphers.
        for char in str_id:
            int_char = ord(char)
            # ensure 3 chars representation for each character
            str_int_char = str(int_char) if int_char > 99 else ('0' + str(int_char))
            id = '{}{}'.format(id, str_int_char)
        return int(id)

    def _decode_generic_id(self, str_id):
        if not isinstance(str_id, str):
            str_id = str(str_id)
        id = ''
        # ascii representation of an alphabet char is always a number
        # of two/three ciphers, stick to three digits
        if len(str_id) % 3 != 0:
            str_id = '0' + str_id

        for idx in range(0, len(str_id), 3):
            char = chr(int(str_id[idx:idx+3]))
            id = '{}{}'.format(id, str(char))
        return id

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            data_raw = json.load(fin)
        return data_raw

    def _create_examples(self, data):
        """Creates examples for the training and dev sets."""
        examples = []
        for data_raw in data['data']:
            article = data_raw['article']
            # encode by default. When input_batching with ids, no string tensor is
            # allowed, ensure that example ids are always numeric.
            for i in range(len(data_raw["answers"])):
                id_str = '{}-{}'.format(data_raw['id'], i)
                example_id = self._encode_generic_id(id_str)
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=example_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer=None,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    # lazy load
    from transformers import PreTrainedTokenizer

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    max_crop = 0
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            # truncate only from context, never from question/ending
            inputs = tokenizer.encode_plus(
                text_a, text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation_strategy='only_first',
                return_overflowing_tokens=True)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length! ({}/{})".format(inputs["num_truncated_tokens"], max_length)
                )
                if inputs["num_truncated_tokens"] > max_crop:
                    max_crop = inputs["num_truncated_tokens"]

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        # if ex_index < 2:
        #     logger.info("*** Example ***")
        #     logger.info("race_id: {}".format(example.example_id))
        #     for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
        #         logger.info("choice: {}".format(choice_idx))
        #         logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
        #         logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
        #         logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
        #         logger.info("label: {}".format(label))

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))

    return features, max_crop


processors = {
    "race": RaceProcessor,
    "swag": SwagProcessor,
    "arc": ArcProcessor,
    'ee': EntranceExamsProcessor,
    'generic': GenericProcessor,
}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4, "ee", 4, "generic", 4}
