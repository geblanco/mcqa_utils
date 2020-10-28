"""Main module."""
import json
import argparse

from pathlib import Path

from mcqa_utils.dataset import Dataset
from mcqa_utils.utils import get_mask_matching_text
from mcqa_utils.threshold import Threshold
from mcqa_utils.metric import metrics_map
from mcqa_utils.evaluate import GenericEvaluator
from mcqa_utils.question_answering import QASystemForMCOffline
from mcqa_utils.answer import (
    apply_threshold_to_answers,
    apply_prob_field_to_answers,
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
        choices=['train', 'dev', 'test'],
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
        '-t', '--threshold', default=None, required=False, type=float,
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
    parser.add_argument(
        '--no_answer_text', type=str, required=False, default=None,
        help='Text of an unaswerable question answer'
    )
    parser.add_argument(
        '-pf', '--probs_field', type=str, required=False, default=None,
        help='Field to use as `probs` field in prediction answers '
        '(default probs, but can be anything parsed in the answer)'
    )
    parser.add_argument(
        '-fm', '--fill_missing', default=None, required=False,
        help='Fill missing answers. Can be filled following a uniform, '
        'random choosing or giving a value for all probs '
        '(uniform/random/value)'
    )
    # ToDo := Add metrics
    args = parser.parse_args()
    if args.nbest_predictions is None and args.predictions is None:
        raise ValueError('You must provide some predictions to evalute!')
    return args


def get_masks_and_prefix(dataset, samples, no_answer_text):
    answer_mask_fn = get_mask_matching_text(no_answer_text, match=False)
    no_answer_mask_fn = get_mask_matching_text(no_answer_text, match=True)
    answer_mask = dataset.find_mask(samples, answer_mask_fn)
    no_answer_mask = dataset.find_mask(samples, no_answer_mask_fn)
    masks = (answer_mask, no_answer_mask)
    prefix = ('has_ans', 'no_has_ans')

    return masks, prefix


def get_results(
    dataset,
    evaluator,
    gold_answers,
    answers,
    masks=None,
    prefixes=None
):
    results = dict()
    if masks is not None and prefixes is not None:
        for mask, prefix in zip(masks, prefixes):
            if sum(mask) > 0:
                gold_reduced = dataset.reduce_by_mask(gold_answers, mask)
                answers_reduced = dataset.reduce_by_mask(answers, mask)
                res = evaluator.evaluate(gold_reduced, answers_reduced)
                results.update(**{prefix: res})

    global_results = evaluator.evaluate(gold_answers, answers)
    results.update(**global_results)
    return results


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

    split = args.split
    no_answer = -1
    metrics = [metrics_map[met]() for met in args.metrics]
    for metric in metrics:
        if metric.needs_no_answer():
            metric.no_answer = no_answer

    dataset = Dataset(data_path=dataset_path, task=args.task)
    qa_system = QASystemForMCOffline(answers_path=results_path)
    evaluator = GenericEvaluator(metrics=metrics)
    threshold = Threshold(evaluator)

    if args.fill_missing is not None:
        qa_system.missing_strategy = args.fill_missing
    # when `no_answer_text` is provided, avg can be optimized with the
    # threshold to answer the option with the text corresponding to
    # not being able to solve the question
    if args.no_answer_text:
        gold_answers = dataset.get_gold_answers(split, with_text_values=True)
        answers, missing = qa_system.get_answers(
            gold_answers,
            with_text_values=True,
            no_answer_text=args.no_answer_text,
        )
        masks, prefix = get_masks_and_prefix(
            dataset, gold_answers, args.no_answer_text
        )
    else:
        gold_answers = dataset.get_gold_answers(split)
        answers, missing = qa_system.get_answers(gold_answers)
        masks = None
        prefix = None

    assert(len(missing) == 0)

    if args.probs_field is not None:
        apply_prob_field_to_answers(answers, args.probs_field)
        # new probs field is not necessary contrained to between 0 and 1
        # search for the lowest and set it as threshold to ensure fair
        # comparison
        min_prob = min([ans.get_min_prob() for ans in answers])
        apply_threshold_to_answers(answers, min_prob - 1.0)

    # get results without threshold mangling
    results_dict = get_results(
        dataset,
        evaluator,
        gold_answers,
        answers,
        masks,
        prefix,
    )

    # get results with requested threshold (if any)
    min_prob_pre_threshold = min([ans.get_min_prob() for ans in answers])
    if args.threshold is not None:
        apply_threshold_to_answers(answers, args.threshold)
        results_dict[f'threshold_{args.threshold}'] = get_results(
            dataset,
            evaluator,
            gold_answers,
            answers,
            masks,
            prefix,
        )

    # find threshold for each requested metric
    if args.find_threshold:
        for metric in metrics:
            if args.threshold:
                # reset threshold to ensure fair comparison
                apply_threshold_to_answers(answers, min_prob_pre_threshold)

            best_threshold = threshold.find_best_threshold(
                metric, gold_answers, answers
            )
            apply_threshold_to_answers(answers, best_threshold)
            threshold_results = get_results(
                dataset,
                evaluator,
                gold_answers,
                answers,
                masks,
                prefix
            )
            threshold_results['threshold'] = best_threshold
            threshold_name = f'{metric.name}_threshold'
            results_dict.update(**{threshold_name: threshold_results})

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
