import os
import json


class QASystem(object):
    def __init__(self, offline: bool = True, answers_path: str = None):
        self.answers = []
        self.offline = offline
        self.answers_path = answers_path
        if offline and answers_path is None:
            raise ValueError(
                'You must provide a path to the answers '
                'if the system is offline!'
            )

    def get_answer(self, example_id):
        if self.offline:
            if example_id not in self.answers:
                raise ValueError('Example not found %r' % example_id)
            else:
                return self.answers[example_id]
        else:
            raise NotImplementedError('You must implement `get_answer` method!')

    def load_predictions(self, path):
        full_path = os.path.abspath(path)
        with open(full_path, 'r') as fstream:
            answers = json.load(fstream)
        return answers


class QASystemForMCOffline(QASystem):

    def __init__(self, answers_path):
        offline = True
        super(QASystemForMCOffline, self).__init__(offline, answers_path)
        raw_answers = self.load_predictions(answers_path)
        self.answers = {}
        for answer in raw_answers:
            self.answers[answer['id']] = answer

    def get_answer(self, example_id):
        answer = super(QASystemForMCOffline, self).get_answer(example_id)
        # in MC nbest predictions, the answer is in the pred_label field
        return answer['pred_label']
