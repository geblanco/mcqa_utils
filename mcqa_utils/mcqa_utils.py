"""Main module."""
import json
import argparse

from pathlib import Path
from functools import partial

from mcqa_utils.dataset import Dataset
from mcqa_utils.utils import label_to_id
from mcqa_utils.threshold import Threshold
from mcqa_utils.metric import metrics_map
from mcqa_utils.evaluate import GenericEvaluator
from mcqa_utils.question_answering import QASystemForMCOffline
from mcqa_utils.answer import (
    input_example_to_answer,
    apply_threshold_to_answers,
)

FLAGS = None

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
        '-s', '--split', default='dev', required=False,
        help='Split to evaluate from the dataset'
    )
    parser.add_argument(
        '-T', '--task', default=None, required=False,
        help='Task to evaluate (default = generic). This '
        'is needed for the dataset processor (see geblanco/mc-transformers)'
    )
    parser.add_argument(
        '-ft', '--find_threshold', action='store_true', required=False,
        help='Perfom threshold search over the answers and apply the metrics'
    )
    parser.add_argument(
        '-t', '--threshold', default=0.0, required=False,
        help='Apply threshold to all answers'
    )
    parser.add_argument(
        '-m', '--metrics', nargs='*', required=True,
        help=f'Metrics to apply (available: {", ".join((metrics_map.keys()))})'
    )
    parser.add_argument(
        '-o', '--output', default=None, required=False,
        help='Whether to put the results (default = stdout)'
    )
    parser.add_argument(
        '--overwrite', action='store_true', required=False,
        help='Overwrite output file (default false)'
    )
    parser.add_argument(
        '--merge', action='store_true', required=False,
        help='Whether to merge output file with previous output'
    )
    # ToDo := Add metrics
    args = parser.parse_args()
    if args.nbest_predictions is None and args.predictions is None:
        raise ValueError('You must provide some predictions to evalute!')
    return args


def answer_mask_fn(mask_cfg, sample):
    mask_text = mask_cfg['text']
    keep_if_found = mask_cfg['match']
    ans_index = label_to_id(sample.label)
    answer = sample.endings[ans_index]
    found = answer.find(mask_text) != -1
    keep = (found and keep_if_found) or (not found and not keep_if_found)
    return keep


def main():
    global FLAGS
    if FLAGS is None:
        FLAGS = parse_flags()
    args = FLAGS
    if args.output is not None:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists() and not args.overwrite and not args.merge:
            raise RuntimeError(
                'Output file already exists!\n'
                'Pass --overwrite or --merge to overcome'
            )
        elif args.merge:
            prev_output = json.load(open(args.output, 'r'))
    else:
        prev_output = None

    dataset_path = args.dataset
    results_path = (
        args.nbest_predictions
        if args.predictions is None
        else args.predictions
    )

    no_answer_text = 'not enough information'
    partial_answer_mask = partial(
        answer_mask_fn,
        {'text': no_answer_text, 'match': False}
    )

    partial_no_answer_mask = partial(
        answer_mask_fn,
        {'text': no_answer_text, 'match': True}
    )

    split = args.split
    no_answer = -1
    metrics = [metrics_map[met]() for met in args.metrics]
    for metric in metrics:
        if metric.needs_no_answer():
            metric.no_answer = no_answer

    dataset = Dataset(data_path=dataset_path, task=args.task)
    data = dataset.get_split(split)
    gold_answers = [input_example_to_answer(ans) for ans in data]
    answer_mask = dataset.find_mask(split, partial_answer_mask)
    no_answer_mask = dataset.find_mask(split, partial_no_answer_mask)

    qa_system = QASystemForMCOffline(answers_path=results_path)
    evaluator = GenericEvaluator(metrics=metrics)
    threshold = Threshold(evaluator)

    answers, missing = qa_system.get_answers(data)
    assert(len(missing) == 0)

    ans_results = evaluator.evaluate(gold_answers, answers, keep=answer_mask)
    if sum(no_answer_mask) == 0:
        # do not evaluate when there are no unanswerable questions
        no_ans_results = {}
    else:
        no_ans_results = evaluator.evaluate(
            gold_answers, answers, keep=no_answer_mask
        )
    if args.threshold > 0.0:
        apply_threshold_to_answers(answers, args.threshold)

    results = evaluator.evaluate(gold_answers, answers)

    results_dict = dict(
        **results,
        has_ans=ans_results,
        no_has_ans=no_ans_results,
    )

    if args.find_threshold:
        best_threshold = threshold.find_best_threshold(
            metrics[0], gold_answers, answers
        )
        apply_threshold_to_answers(answers, best_threshold)
        threshold_results = evaluator.evaluate(gold_answers, answers)
        threshold_results['threshold'] = best_threshold
        results_dict.update(best_threshold=threshold_results)

    results_str = json.dumps(obj=results_dict, indent=2) + '\n'
    if args.output is None:
        print(results_str)
    else:
        if args.merge and prev_output is not None:
            prev_output.update(**results_dict)
            results_str = json.dumps(obj=prev_output, indent=2) + '\n'
        
        with open(args.output, 'w') as fout:
            fout.write(results_str)


if __name__ == '__main__':
    main()
