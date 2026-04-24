from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def reduce(to_reduce: Tensor, reduction: str) -> Tensor:
    """Reduces a given tensor by a given reduction method.

    Args:
        to_reduce: the tensor, which shall be reduced
        reduction:  a string specifying the reduction method ('elementwise_mean', 'none', 'sum')

    Return:
        reduced Tensor

    Raise:
        ValueError if an invalid reduction parameter was given
    """
    pass


def class_reduce(
    num: Tensor, denom: Tensor, weights: Tensor, class_reduction: str = "none"
) -> Tensor:
    """
    Function used to reduce classification metrics of the form `num / denom * weights`.
    For example for calculating standard accuracy the num would be number of
    true positives per class, denom would be the support per class, and weights
    would be a tensor of 1s

    Args:
        num: numerator tensor
        denom: denominator tensor
        weights: weights for each class
        class_reduction: reduction method for multiclass problems

            - ``'micro'``: calculate metrics globally (default)
            - ``'macro'``: calculate metrics for each label, and find their unweighted mean.
            - ``'weighted'``: calculate metrics for each label, and find their weighted mean.
            - ``'none'`` or ``None``: returns calculated metric per class

    Raises:
        ValueError:
            If ``class_reduction`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"`` or ``None``.

    """
    pass


def _simple_gather_all_tensors(
    result: Tensor, group: Any, world_size: int
) -> List[Tensor]:
    pass


def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    pass
