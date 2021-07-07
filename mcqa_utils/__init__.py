"""Top-level package for mcqa_utils."""

__author__ = """Guillermo E. Blanco"""
__email__ = 'geblanco@lsi.uned.es'
__version__ = '0.2.3'

# flake8: noqa

from .dataset import Dataset
from .evaluate import GenericEvaluator
from .question_answering import QASystemForMCOffline
from .threshold import Threshold

from .answer import (
    Answer,
    parse_answer,
    apply_threshold_to_answers,
    apply_no_answer,
)
from .metric import (
    C_at_1,
    F1,
    Average,
    metrics_map,
)

from .utils import (
    get_mask_matching_text,
    answer_mask_fn
)
