"""Main module."""
import json
import argparse

from pathlib import Path
from collections import defaultdict

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


def parse_utility_fn_str(uf_str):
    parts = uf_str.split()
    assert(len(parts) % 3 == 0)
    return [
        [float(uf) for uf in parts[i:i + 3]]
        for i in range(0, len(parts), 3)
    ]


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
        '-i', '--info', action='store_true',
        help="Just print info about the dataset"
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
        '-m', '--metrics', nargs='*', required=False, default=[],
        help=f'Metrics to apply (available: {", ".join((metrics_map.keys()))})'
    )
    parser.add_argument(
        "-uf", "--utility_function", nargs=3,
        type=float, default=[], action="append",
        help="Weights for the utility function (implies -m utility_function)."
        " Tuples of three values are required [unanswered, incorrect, correct"
        "]. If multiple values are provided, several utility functions will be"
        " applied"
    )
    parser.add_argument(
        "-ufs", "--utility_function_str", type=str, default="",
        help="Weights for the utility function (passed as string) "
        "(implies -m utility_function). Tuples of three values are required "
        "[unanswered, incorrect, correct]. If multiple values are provided, "
        "several utility functions will be applied"
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
    parser.add_argument(
        "--save_mlflow", action="store_true",
        help="Stores the given metrics in mlflow (requires package installed)"
    )
    args = parser.parse_args()
    args.utility_function.extend(
        parse_utility_fn_str(args.utility_function_str)
    )
    # uniq
    args.metrics = list(set(args.metrics))
    if not args.info and (
        len(args.metrics) == 0 and len(args.utility_function) == 0
    ):
        raise ValueError(
            "If not printing dataset info, you must request at "
            "least one metric!"
        )
    elif (
        not args.info and
        (args.nbest_predictions is None and args.predictions is None)
    ):
        raise ValueError('You must provide some predictions to evalute!')

    # delete metrics with non-default values, will be created separately
    if len(args.utility_function) > 0 and "utility_function" in args.metrics:
        del args.metrics[args.metrics.index("utility_function")]

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


def get_dataset_split(dataset, split, with_text_values=False):
    gold_answers = None
    try:
        gold_answers = dataset.get_gold_answers(
            split, with_text_values=with_text_values
        )
    except Exception:
        pass
    finally:
        return gold_answers


def print_dataset_stats(args):
    # sample table
    # |                 | train | dev  | test |
    # | # questions     | 14000 | 4000 | 5000 |
    # | # unanswerable  | 200   | 100  | 100  |
    dataset = Dataset(data_path=args.dataset, task=args.task)
    splits = []
    results = defaultdict(list)
    for split in dataset.splits:
        gold_answers = get_dataset_split(
            dataset, split, with_text_values=args.no_answer_text is not None
        )
        if gold_answers is None:
            print(f'Split {split} not found in dataset')
            continue

        results['# questions'].append(len(gold_answers))
        if args.no_answer_text:
            no_answer_mask_fn = get_mask_matching_text(
                args.no_answer_text, match=True
            )
            no_answer_mask = dataset.find_mask(gold_answers, no_answer_mask_fn)
            gold_reduced = dataset.reduce_by_mask(gold_answers, no_answer_mask)
            results['# unanswerable'].append(len(gold_reduced))

        splits.append(split)

    start_width = max([len(str(key)) for key in results.keys()])
    width = max([
        len(str(item)) for values in results.values()
        for item in values
    ] + [len(sp) for sp in splits])

    head_fmt = '{:>{mwidth}s}' + '\t{:>{width}s}' * len(splits)
    results_str = head_fmt.format(
        "", *splits, mwidth=start_width, width=width
    ) + '\n'
    for key, values in results.items():
        str_values = [str(val) for val in values]
        results_str += head_fmt.format(
            key, *str_values, mwidth=start_width, width=width
        ) + '\n'

    print(results_str)


def mcqa(args):
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
    if len(args.utility_function) > 0:
        for uf in args.utility_function:
            met = metrics_map["utility_function"]()
            met.utility = uf
            metrics.append(met)

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

    if args.save_mlflow:
        import mlflow
        mlflow.log_metrics(results_dict)


def main():
    args = parse_flags()
    if args.info:
        print_dataset_stats(args)
    else:
        mcqa(args)


if __name__ == '__main__':
    main()
