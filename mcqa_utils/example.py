import argparse

from functools import partial
from mcqa_utils.metric import C_at_1
from mcqa_utils.dataset import Dataset
from mcqa_utils.evaluator import GenericEvaluator
from mcqa_utils.question_answering import QASystemForMCOffline


def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--predictions', required=False, default=None,
        help='Predictions from model'
    )
    parser.add_argument(
        '-n', '--nbest_predictions', required=False, default=None,
        help='Nbest predictions from model'
    )
    parser.add_argument(
        '-d', '--dataset', required=True,
        help='Directory where the dataset is stored'
    )
    parser.add_argument(
        '-t', '--task', default=None, required=False,
        help='Task to evaluate (default = generic). This '
        'is needed for the dataset processor (see geblanco/mc-transformers)'
    )
    args = parser.parse_args()
    if args.nbest_predictions is None and args.predictions is None:
        raise ValueError('You must provide some predictions to evalute!')
    return args


def answer_mask_fn(equal, sample):
    ans_index = int(sample.label)
    answer = sample.endings[ans_index]
    found = answer.find('not enough information')
    keep = (found != -1 and equal) or (found == -1 and not equal)
    return keep


def main(args):
    dataset_path = args.dataset
    results_path = (
        args.nbest_predictions
        if args.predictions is None
        else args.predictions
    )

    dataset = Dataset(data_path=dataset_path, task=args.task)
    evaluator = GenericEvaluator(metric=C_at_1(no_answer=-1))

    split = 'dev'

    answer_mask = dataset.find_mask(split, partial(answer_mask_fn, False))
    no_answer_mask = dataset.find_mask(split, partial(answer_mask_fn, True))

    qa_system = QASystemForMCOffline(answers_path=results_path)
    ans_results = evaluator.evaluate(
        qa_system, dataset,
        split, keep=answer_mask
    )
    no_ans_results = evaluator.evaluate(
        qa_system, dataset,
        split, keep=no_answer_mask
    )
    results = evaluator.evaluate(qa_system, dataset, split)
    print(f'Results over all dataset: {results}')
    print(f'Results answered questions: {ans_results}')
    print(f'Results (un)answered questions: {no_ans_results}')


if __name__ == '__main__':
    main(parse_flags())
