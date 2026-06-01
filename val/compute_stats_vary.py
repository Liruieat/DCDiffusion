#!/usr/bin/env python3
"""
Nuclear Segmentation and Classification Metrics Evaluation Script

This script calculates comprehensive evaluation metrics for nuclear segmentation
and classification tasks. It implements multiple metrics including Dice coefficient,
AJI (Aggregated Jaccard Index), AJI+, PQ (Panoptic Quality), and type-specific metrics.

Supported Metrics:
- Instance-level: Dice, AJI, AJI+, PQ (DQ, SQ)
- Type-level: F1 score for each nuclear type
- Multi-type: mDice, mAJI, mPQ across different nuclear categories

Usage:
    python compute_stats_vary.py --mode instance --n_pred_dir <pred_dir> --n_true_dir <gt_dir>
    python compute_stats_vary.py --mode type --n_pred_dir <pred_dir> --n_true_dir <gt_dir>

Author: DCDiffusion Framework
"""

import argparse
import glob
import os

import numpy as np
import torch
import scipy
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt

# Note: Set GPU device if needed for your environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import metric calculation utilities
from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_pq,
    remap_label,
    pair_coordinates,
    get_fast_pq_get_list,
    get_fast_aji_get_list,
    get_fast_dice_2_get_list,
    get_dice_1_get_list,
    get_fast_aji_plus_get_list,
)

from metrics.metrics import (
    aji,
    pq,
    mpq,
    remap_label_fast,
    aji_list,
    aji_plus_list,
    pq_list,
    mpq_list,
)

from metrics.inst_metrics import (
    pre_eval_aji,
    pre_eval_to_aji,
    pre_eval_pq,
    pre_eval_to_pq,
)


def run_nuclei_type_stat(pred_dir, true_dir, type_uid_list=None, exhaustive=True):
    """
    Calculate type classification metrics for nuclear segmentation results.

    This function computes F1 scores for nuclear type classification at both
    instance and type levels. It handles paired and unpaired nuclei between
    predictions and ground truth.

    Args:
        pred_dir (str): Directory containing prediction .mat files
        true_dir (str): Directory containing ground truth .mat files
        type_uid_list (list, optional): List of type IDs to evaluate.
                                       If None, uses all types in GT.
        exhaustive (bool): Whether GT is exhaustively annotated for types.
                          Set to False for partially annotated datasets.

    Returns:
        None: Results are printed and saved to metrics_miji.txt

    .mat File Requirements:
        - inst_centroid: Nx2 array of centroid coordinates (X, Y)
        - inst_type: Nx1 array of type labels
        - Arrays must be aligned by index

    Example:
        >>> run_nuclei_type_stat('/path/to/preds', '/path/to/gt',
        ...                      type_uid_list=[1,2,3,4], exhaustive=True)
    """

    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        """Calculate F1 score for a specific nuclear type."""
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        # Calculate true positives, true negatives, false positives, false negatives
        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        # Calculate weighted F1 score
        f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
        )
        return f1_type

    def compute_f1_for_multiple_types(paired_true, paired_pred, unpaired_true, unpaired_pred, w, type_uid_list, exhaustive=False):
        """Calculate aggregated F1 score across multiple nuclear types."""
        processed_mask = np.zeros_like(paired_true, dtype=bool)

        total_tp_dt = 0
        total_tn_dt = 0
        total_fp_dt = 0
        total_fn_dt = 0
        total_fp_d = 0
        total_fn_d = 0

        for type_id in type_uid_list:
            # Select samples relevant to current type
            type_samples = (paired_true == type_id) | (paired_pred == type_id)
            type_samples = type_samples & ~processed_mask

            paired_true_filtered = paired_true[type_samples]
            paired_pred_filtered = paired_pred[type_samples]

            # Calculate confusion matrix elements
            tp_dt = ((paired_true_filtered == type_id) & (paired_pred_filtered == type_id)).sum()
            tn_dt = ((paired_true_filtered != type_id) & (paired_pred_filtered != type_id)).sum()
            fp_dt = ((paired_true_filtered != type_id) & (paired_pred_filtered == type_id)).sum()
            fn_dt = ((paired_true_filtered == type_id) & (paired_pred_filtered != type_id)).sum()

            if not exhaustive:
                ignore = (paired_true_filtered == -1).sum()
                fp_dt -= ignore

            fp_d = (unpaired_pred == type_id).sum()
            fn_d = (unpaired_true == type_id).sum()

            # Update totals
            total_tp_dt += tp_dt
            total_tn_dt += tn_dt
            total_fp_dt += fp_dt
            total_fn_dt += fn_dt
            total_fp_d += fp_d
            total_fn_d += fn_d

            processed_mask[type_samples] = True

        # Calculate final F1 score
        f1_type = (2 * (total_tp_dt + total_tn_dt)) / (
            2 * (total_tp_dt + total_tn_dt)
            + w[0] * total_fp_dt
            + w[1] * total_fn_dt
            + w[2] * total_fp_d
            + w[3] * total_fn_d
        )

        return f1_type

    # Get all prediction files
    file_list = glob.glob("%s/*%s" % (pred_dir, "*.mat"))
    file_list.sort()  # Ensure consistent ordering

    # Initialize containers for paired/unpaired nuclei
    paired_all = []
    unpaired_true_all = []
    unpaired_pred_all = []
    true_inst_type_all = []
    pred_inst_type_all = []

    # Process each file
    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        # Load ground truth data
        true_info = sio.loadmat(os.path.join(true_dir, basename + ".mat"))
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]).astype("int32").flatten()

        # Load prediction data
        pred_info = sio.loadmat(os.path.join(pred_dir, basename + ".mat"))
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # Pair nuclei between prediction and ground truth
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, 12
        )

        # Calculate index offsets for aggregation
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )

        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # Aggregate paired and unpaired nuclei
        if paired.shape[0] != 0:
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    # Concatenate all results
    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)

    true_inst_type_all = np.concatenate([np.squeeze(arr) for arr in true_inst_type_all], axis=0)
    pred_inst_type_all = np.concatenate([np.squeeze(arr) for arr in pred_inst_type_all], axis=0)

    # Extract paired and unpaired types
    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    # Calculate overall detection and classification metrics
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    # Calculate type-specific F1 scores
    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    # Calculate weighted multi-type F1
    weight_type_f1 = compute_f1_for_multiple_types(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            w,
            type_uid_list
    )
    results_list.append(weight_type_f1)

    # Print and save results
    np.set_printoptions(formatter={"float": "{: 0.6f}".format})
    print(np.array(results_list))

    metrics_avg = np.array(results_list)
    metrics_avg_str = ['{:.6f}'.format(x) for x in metrics_avg]
    with open(f'{os.path.dirname(pred_dir)}/metrics_miji.txt', 'a') as f:
        f.write('\t'.join(metrics_avg_str))
        f.write('\n')
    return


def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=True, ext=".mat"):
    """
    Calculate basic instance segmentation metrics.

    This is a simplified version that computes Dice, AJI, AJI+, and PQ metrics
    without type-specific analysis.

    Args:
        pred_dir (str): Prediction directory
        true_dir (str): Ground truth directory
        print_img_stats (bool): Whether to print per-image statistics
        ext (str): File extension to process

    Returns:
        list: Array of computed metrics
    """
    print(f"Processing directory: {pred_dir}")

    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()

    # Initialize metric accumulators
    metrics = [[], [], [], [], [], [], []]
    pq_list = [0, 0, 0, 0]
    fast_aji_list = [0, 0]
    fast_aji_plus_list = [0, 0]
    dice_1_list = [0, 0]

    # Process each image
    for filename in tqdm(file_list):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        # Load instance maps
        true = scipy.io.loadmat(os.path.join(true_dir, basename + ".mat"))["inst_map"].astype("int32")
        pred = scipy.io.loadmat(os.path.join(pred_dir, basename + ".mat"))["inst_map"].astype("int32")

        # Ensure contiguous instance numbering
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        # Calculate metrics
        pq_pre_list = get_fast_pq_get_list(true, pred, match_iou=0.5)
        pq_list = elementwise_addition(pq_list, pq_pre_list)

        fast_aji_pre_list = get_fast_aji_get_list(true, pred)
        fast_aji_list = elementwise_addition(fast_aji_list, fast_aji_pre_list)

        fast_aji_plus_pre_list = get_fast_aji_plus_get_list(true, pred)
        fast_aji_plus_list = elementwise_addition(fast_aji_plus_list, fast_aji_plus_pre_list)

        dice_1_pre_list = get_dice_1_get_list(true, pred)
        dice_1_list = elementwise_addition(dice_1_list, dice_1_pre_list)

    # Calculate final metrics
    metrics[0].append(2 * calculate_ratio(dice_1_list))
    metrics[1].append(calculate_ratio(fast_aji_list))
    metrics[2].append(calculate_ratio(fast_aji_plus_list))

    pq_result = calculate_pq(pq_list)
    metrics[3].append(pq_result[0])  # dq
    metrics[4].append(pq_result[1])  # sq
    metrics[5].append(pq_result[2])  # pq

    # Convert to array and save
    metrics_array = np.array(metrics)
    print(metrics_array)

    flat_metrics = metrics_array.flatten()
    with open(f'{os.path.dirname(pred_dir)}/metrics_miji.txt', 'a') as f:
        np.savetxt(f, [flat_metrics], fmt='%.5f', delimiter='\t')

    return metrics


def elementwise_addition(list1, list2):
    """Perform element-wise addition of two lists.

    Args:
        list1 (list): First list of numbers
        list2 (list): Second list of numbers (must be same length as list1)

    Returns:
        list: Element-wise sum of list1 and list2

    Raises:
        ValueError: If lists have different lengths
    """
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")
    return [x + y for x, y in zip(list1, list2)]


def calculate_ratio(lst):
    """Calculate ratio from a two-element list.

    Args:
        lst (list): List with exactly 2 numeric elements [numerator, denominator]

    Returns:
        float: lst[0] / lst[1]

    Raises:
        ValueError: If list has fewer than 2 elements
        ZeroDivisionError: If denominator is zero
    """
    if len(lst) < 2:
        raise ValueError("The list must have at least two elements.")

    if lst[1] == 0:
        raise ZeroDivisionError("The second element of the list cannot be zero.")

    return lst[0] / lst[1]


def calculate_pq(metrics_list):
    """Calculate Panoptic Quality (PQ) metrics.

    PQ = SQ * DQ, where:
    - DQ (Detection Quality) = TP / (TP + 0.5*FP + 0.5*FN)
    - SQ (Segmentation Quality) = sum(IoUs) / TP

    Args:
        metrics_list (list): [TP, FP, FN, sum_of_IoUs]

    Returns:
        list: [DQ, SQ, PQ]
    """
    if len(metrics_list) != 4:
        raise ValueError("The list must contain exactly four elements: [tp, fp, fn, paired_iou_sum]")

    # Convert torch tensors to numpy arrays if needed
    converted_data = [
        [value.item() if isinstance(value, torch.Tensor) else value for value in row]
        if isinstance(row, (list, tuple)) else row
        for row in metrics_list
    ]
    metrics_list = converted_data
    tp, fp, fn, paired_iou_sum = metrics_list

    # Sum if lists
    tp = sum(tp) if isinstance(tp, list) else tp
    fp = sum(fp) if isinstance(fp, list) else fp
    fn = sum(fn) if isinstance(fn, list) else fn
    paired_iou_sum = sum(paired_iou_sum) if isinstance(paired_iou_sum, list) else paired_iou_sum

    # Calculate DQ (Detection Quality)
    if (tp + 0.5 * fp + 0.5 * fn) == 0:
        dq = 0
    else:
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)

    # Calculate SQ (Segmentation Quality)
    sq = paired_iou_sum / (tp + 1.0e-6)

    # Calculate PQ
    dq_sq = dq * sq

    return [dq, sq, dq_sq]


def m_ca(true_type_map, true_type_maps, pred_type_map, pred_type_maps, method_flag='dice'):
    """
    Compute metrics (Dice/AJI/AJI+/PQ) for each nuclear type separately.

    This function separates different nuclear types and computes metrics
    for each type independently.

    Args:
        true_type_map: 2D array of ground truth types
        true_type_maps: Dict of type-specific instance maps for GT
        pred_type_map: 2D array of predicted types
        pred_type_maps: Dict of type-specific instance maps for predictions
        method_flag: Metric type ('dice', 'aji', 'aji_plus', 'pq')

    Returns:
        dict: Type-specific metric results
    """
    total_results = {}

    # Process each type
    for t in range(int(min(torch.min(pred_type_map).item(), torch.min(true_type_map).item())),
                   int(max(torch.max(pred_type_map).item(), torch.max(true_type_map).item())) + 1):

        # Get instance maps for current type
        if np.any(true_type_maps.get(t)) and np.any(pred_type_maps.get(t)):
            true_type_instance_map = remap_label_fast(true_type_maps.get(t, np.zeros_like(true_type_map)))
            pred_type_instance_map = remap_label_fast(pred_type_maps.get(t, np.zeros_like(pred_type_map)))
        elif np.any(true_type_maps.get(t)) or np.any(pred_type_maps.get(t)):
            true_type_instance_map = remap_label_fast(true_type_maps.get(t, np.zeros_like(true_type_map)))
            pred_type_instance_map = remap_label_fast(pred_type_maps.get(t, np.zeros_like(pred_type_map)))
        else:
            continue

        # Calculate appropriate metric for this type
        if method_flag == 'dice' and t != 0:
            total_results[t] = get_dice_1_get_list(true_type_instance_map, pred_type_instance_map)
        elif method_flag == 'aji':
            total_results[t] = aji_list(true_type_instance_map, pred_type_instance_map)
        elif method_flag == 'aji_plus':
            total_results[t] = aji_plus_list(true_type_instance_map, pred_type_instance_map)
        elif method_flag == 'pq':
            total_results[t] = pq_list(true_type_instance_map, pred_type_instance_map)

    if len(total_results) == 0:
        print(0)
    return total_results


def calculate_m(total_results, method_flag=''):
    """
    Calculate aggregated metrics across all nuclear types.

    Args:
        total_results (dict): Type-specific results from m_ca()
        method_flag (str): Metric type ('dice', 'aji', 'aji_plus', 'pq')

    Returns:
        tuple: (average_value, list_value, account_list)
            - average_value: Mean across types
            - list_value: Total aggregated value
            - account_list: Accumulated components
    """
    # Initialize
    try:
        first_value = next(iter(total_results.values()))
    except:
        first_value = []
        return 0, 0, [0]

    account_list = [0] * len(first_value)

    # Calculate totals
    total_sum = 0
    total_dq = 0
    total_sq = 0

    for t, result_list in total_results.items():
        if method_flag == 'dice' and t != 0:
            for i, value in enumerate(result_list):
                account_list[i] += value
            total_sum += 2 * result_list[0] / (result_list[1] + 1e-12)

        elif method_flag == 'aji' or method_flag == 'aji_plus':
            for i, value in enumerate(result_list):
                account_list[i] += value
            total_sum += result_list[0] / (result_list[1] + 1e-12)

        elif method_flag == 'pq':
            for i, value in enumerate(result_list):
                account_list[i] += value

            pq_result = calculate_pq(result_list)
            total_dq += pq_result[0]
            total_sq += pq_result[1]
            total_sum += pq_result[2]

    # Calculate averages
    average_value = total_sum / len(total_results)
    if method_flag == 'pq':
        average_value = [total_dq / len(total_results), total_sq / len(total_results), average_value]

    if method_flag == 'dice':
        list_value = 2 * account_list[0] / (account_list[1] + 1e-12)
    elif method_flag == 'aji' or method_flag == 'aji_plus':
        list_value = account_list[0] / (account_list[1] + 1e-12)
    elif method_flag == 'pq':
        list_value = calculate_pq(account_list)

    return average_value, list_value, account_list


def create_type_map(inst_map, inst_type, max_type, filename):
    """
    Create type maps from instance maps and type labels.

    This function converts instance maps to type-specific maps for
    multi-type nuclear analysis.

    Args:
        inst_map: 2D instance segmentation map
        inst_type: 1D array of type labels for each instance
        max_type: Maximum type ID
        filename: Image identifier (for debugging)

    Returns:
        tuple: (type_map, type_maps)
            - type_map: 2D array with pixels labeled by type
            - type_maps: Dict of type-specific instance maps
    """
    # Create overall type map
    type_map = np.zeros_like(inst_map)
    type_maps = {}

    # Get unique instance IDs (excluding background)
    unique_instances = np.unique(inst_map)
    unique_instances = unique_instances[unique_instances > 0]

    # Create overall type map
    for i, instance in enumerate(unique_instances):
        type_id = inst_type[i]
        type_map[inst_map == instance] = type_id

    # Create separate instance map for each type
    for t in range(1, max_type + 1):
        type_map_for_type = np.zeros_like(inst_map)
        current_instance_id = 1

        for i, instance in enumerate(unique_instances):
            type_id = inst_type[i]
            if type_id == t:
                type_map_for_type[inst_map == instance] = current_instance_id
                current_instance_id += 1

        type_maps[t] = type_map_for_type

    return type_map, type_maps


def sum_dicts(*dicts):
    """Sum values across multiple dictionaries with the same keys.

    Args:
        *dicts: Variable number of dictionaries to sum

    Returns:
        dict: Dictionary with summed values

    Raises:
        ValueError: If dict values have inconsistent list lengths
    """
    if not dicts:
        return {}

    # Get all unique keys
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())

    result = {}

    for key in all_keys:
        dicts_with_key = [d for d in dicts if key in d]

        if not dicts_with_key:
            continue

        # Check consistency
        list_length = len(dicts_with_key[0][key])
        for d in dicts_with_key:
            if len(d[key]) != list_length:
                raise ValueError(f"Key {key} has inconsistent list lengths across dictionaries")

        # Sum values
        sum_list = [0] * list_length
        for d in dicts_with_key:
            for i in range(list_length):
                sum_list[i] += d[key][i]

        result[key] = sum_list

    return result


def run_nuclei_fast_instance_(pred_dir, true_dir, print_img_stats=True, ext=".mat"):
    """
    Comprehensive instance and type-specific metrics evaluation.

    This is the main evaluation function that calculates:
    - Binary metrics: Dice, AJI, AJI+, PQ
    - Multi-type metrics: mDice, mAJI, mPQ
    - Single-type metrics: Per-type breakdown

    Args:
        pred_dir (str): Prediction directory
        true_dir (str): Ground truth directory
        print_img_stats (bool): Print per-image statistics
        ext (str): File extension

    Returns:
        list: Complete metrics array
    """
    print(f"Processing directory: {pred_dir}")

    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()

    # Initialize metric accumulators
    metrics = [[], [], [], [], [], [], [], [], [], []]
    fast_pq_list = [0, 0, 0, 0]
    fast_aji_list = [0, 0]
    fast_aji_plus_list = [0, 0]
    m_dice_1_list = [0, 0]
    mpq_l = [0, 0, 0, 0]
    m_aji_list = [0, 0]
    m_pq_list = [0, 0, 0, 0]

    sing_dice_list = {}
    sing_aji_list = {}
    sing_pq_list = {}

    # Process each image
    for filename in tqdm(file_list):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        # Load ground truth
        true = scipy.io.loadmat(os.path.join(true_dir, basename + ".mat"))["inst_map"].astype("int32")
        true_inst_type = scipy.io.loadmat(os.path.join(true_dir, basename + ".mat"))["inst_type"].flatten()
        true_centroid = scipy.io.loadmat(os.path.join(true_dir, basename + ".mat"))["inst_centroid"].astype("float32")

        try:
            true_type_map, true_type_maps = create_type_map(true, true_inst_type, 5, filename)

            # Load predictions
            pred = scipy.io.loadmat(os.path.join(pred_dir, basename + ".mat"))["inst_map"].astype("int32")
            pred_inst_type = scipy.io.loadmat(os.path.join(pred_dir, basename + ".mat"))["inst_type"].flatten()
            pred_centroid = scipy.io.loadmat(os.path.join(pred_dir, basename + ".mat"))["inst_centroid"].astype("float32")

            # Note: Coordinate swapping if needed for your data format
            # pred_centroid = pred_centroid[:, [1, 0]]
            pred_type_map, pred_type_maps = create_type_map(pred, pred_inst_type, 5, filename)
        except:
            print(f"Error processing {filename}")
            continue

        # Ensure contiguous numbering
        pred = remap_label_fast(pred)
        true = remap_label_fast(true)

        # Convert to tensors if needed
        if type(true_type_map) != torch.Tensor:
            true_type_map = torch.from_numpy(true_type_map)
        if type(pred_type_map) != torch.Tensor:
            pred_type_map = torch.from_numpy(pred_type_map)

        # Calculate multi-type metrics
        mu_dice = m_ca(true_type_map, true_type_maps, pred_type_map, pred_type_maps, 'dice')
        avg_dice_value, list_dice_value, dice_1_list_process = calculate_m(mu_dice, 'dice')
        m_dice_1_list = elementwise_addition(m_dice_1_list, dice_1_list_process)

        if len(sing_dice_list) == 0:
            sing_dice_list = mu_dice
        else:
            sing_dice_list = sum_dicts(sing_dice_list, mu_dice)

        mu_aji = m_ca(true_type_map, true_type_maps, pred_type_map, pred_type_maps, 'aji')
        avg_aji_value, list_aji_value, m_aji_list_process = calculate_m(mu_aji, 'aji')
        m_aji_list = elementwise_addition(m_aji_list, m_aji_list_process)

        if len(sing_aji_list) == 0:
            sing_aji_list = mu_aji
        else:
            sing_aji_list = sum_dicts(sing_aji_list, mu_aji)

        mu_pq = m_ca(true_type_map, true_type_maps, pred_type_map, pred_type_maps, 'pq')
        avg_pq_value, list_pq_value, m_pq_list_process = calculate_m(mu_pq, 'pq')
        m_pq_list = elementwise_addition(m_pq_list, m_pq_list_process)

        if len(sing_pq_list) == 0:
            sing_pq_list = mu_pq
        else:
            sing_pq_list = sum_dicts(sing_pq_list, mu_pq)

        # Calculate binary metrics
        fast_pq = pq_list(true, pred, match_iou=0.5)
        fast_aji_pre_list = aji_list(true, pred)
        dice_1_pre_list = get_dice_1_get_list(true, pred)
        fast_aji_plus_pre_list = aji_plus_list(true, pred)
        mpq_ll = mpq_list(true, true_type_map, pred, pred_type_map)

        # Accumulate metrics
        metrics[0].append(2 * calculate_ratio(dice_1_pre_list))
        metrics[1].append(2 * calculate_ratio(dice_1_list_process))
        metrics[2].append(calculate_ratio(fast_aji_pre_list))
        metrics[3].append(calculate_ratio(fast_aji_plus_pre_list))

        pq_result = calculate_pq(fast_pq)
        metrics[4].append(pq_result[0])  # b_dq
        metrics[5].append(pq_result[1].cpu().numpy())  # b_sq
        metrics[6].append(pq_result[2].cpu().numpy())  # b_pq

        pq_result = calculate_pq(mpq_ll)
        metrics[7].append(pq_result[0])  # m_dq_avg
        metrics[8].append(pq_result[1].cpu().numpy())  # m_sq_avg
        metrics[9].append(pq_result[2].cpu().numpy())  # m_pq_avg

    # Calculate final metrics
    metrics_array = np.array(metrics)
    metrics_avg = np.mean(metrics_array, axis=-1)

    m_metrics = [[], [], [], [], []]
    m_metrics[0].append(2 * calculate_ratio(dice_1_list_process))
    m_metrics[1].append(calculate_ratio(m_aji_list))

    pq_result = calculate_pq(m_pq_list)
    m_metrics[2].append(pq_result[0])
    m_metrics[3].append(pq_result[1].cpu().numpy())
    m_metrics[4].append(pq_result[2].cpu().numpy())

    m_metrics_array = np.array(m_metrics)

    # Calculate single-type metrics
    single_metrics = {}
    for key in sing_dice_list.keys():
        single_metrics[key] = [[], [], [], [], []]

    for key in sing_dice_list.keys():
        single_metrics[key][0].append(2 * calculate_ratio(sing_dice_list[key]))
        single_metrics[key][1].append(calculate_ratio(sing_aji_list[key]))

        pq_result = calculate_pq(sing_pq_list[key])
        single_metrics[key][2].append(pq_result[0])
        single_metrics[key][3].append(pq_result[1].cpu().numpy())
        single_metrics[key][4].append(pq_result[2].cpu().numpy())

    # Save results to file
    flat_metrics = metrics_avg.flatten()
    flat_m_metrics_array = m_metrics_array.flatten()

    with open(f'{os.path.dirname(pred_dir)}/metrics_miji.txt', 'a') as f:
        f.write("dice, dice1_list, aji, aji_plus, b_dq, b_sq, b_pq, m_dq_avg, m_sq_avg, m_pq_avg\n")
        np.savetxt(f, [flat_metrics], fmt='%.5f', delimiter='\t')
        f.write("m_dice, m_aji, m_dq_list, m_sq_list, m_pq_list\n")
        np.savetxt(f, [flat_m_metrics_array], fmt='%.5f', delimiter='\t')

        f.write("\n=== Single Type Metrics ===\n")
        f.write("Type\tDice\tAJI\tDQ\tSQ\tPQ\n")
        for key in sorted(single_metrics.keys()):
            metrics = single_metrics[key]
            dice_val = metrics[0][0] if metrics[0] else 0
            aji_val = metrics[1][0] if metrics[1] else 0
            dq_val = metrics[2][0] if metrics[2] else 0
            sq_val = metrics[3][0] if metrics[3] else 0
            pq_val = metrics[4][0] if metrics[4] else 0

            f.write(f"{key}\t{dice_val:.5f}\t{aji_val:.5f}\t{dq_val:.5f}\t{sq_val:.5f}\t{pq_val:.5f}\n")

    return metrics


def run_nuclei_fast_instance_single(pred_dir, true_dir, print_img_stats=True, ext=".mat"):
    """
    Simplified comprehensive evaluation without single-type breakdown.

    Similar to run_nuclei_fast_instance_ but excludes per-type detailed analysis.

    Args:
        pred_dir (str): Prediction directory
        true_dir (str): Ground truth directory
        print_img_stats (bool): Print per-image statistics
        ext (str): File extension

    Returns:
        list: Metrics array
    """
    print(f"Processing directory: {pred_dir}")

    file_list = glob.glob("%s/*%s" % (pred_dir, ext))
    file_list.sort()

    metrics = [[], [], [], [], [], [], [], [], [], []]
    fast_pq_list = [0, 0, 0, 0]
    fast_aji_list = [0, 0]
    fast_aji_plus_list = [0, 0]
    m_dice_1_list = [0, 0]
    mpq_l = [0, 0, 0, 0]
    m_aji_list = [0, 0]
    m_pq_list = [0, 0, 0, 0]

    # Process each image
    for filename in tqdm(file_list):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]

        # Load data
        true = scipy.io.loadmat(os.path.join(true_dir, basename + ".mat"))["inst_map"].astype("int32")
        true_inst_type = scipy.io.loadmat(os.path.join(true_dir, basename + ".mat"))["inst_type"].flatten()
        true_centroid = scipy.io.loadmat(os.path.join(true_dir, basename + ".mat"))["inst_centroid"].astype("float32")

        try:
            true_type_map, true_type_maps = create_type_map(true, true_inst_type, 5, filename)

            pred = scipy.io.loadmat(os.path.join(pred_dir, basename + ".mat"))["inst_map"].astype("int32")
            pred_inst_type = scipy.io.loadmat(os.path.join(pred_dir, basename + ".mat"))["inst_type"].flatten()
            pred_centroid = scipy.io.loadmat(os.path.join(pred_dir, basename + ".mat"))["inst_centroid"].astype("float32")

            # Note: Coordinate swapping if needed
            # pred_centroid = pred_centroid[:, [1, 0]]
            pred_type_map, pred_type_maps = create_type_map(pred, pred_inst_type, 5, filename)
        except:
            print(f"Error processing {filename}")
            continue

        # Process maps
        pred = remap_label_fast(pred)
        true = remap_label_fast(true)

        if type(true_type_map) != torch.Tensor:
            true_type_map = torch.from_numpy(true_type_map)
        if type(pred_type_map) != torch.Tensor:
            pred_type_map = torch.from_numpy(pred_type_map)

        # Calculate metrics
        mu_dice = m_ca(true_type_map, true_type_maps, pred_type_map, pred_type_maps, 'dice')
        avg_dice_value, list_dice_value, dice_1_list_process = calculate_m(mu_dice, 'dice')
        m_dice_1_list = elementwise_addition(m_dice_1_list, dice_1_list_process)

        mu_aji = m_ca(true_type_map, true_type_maps, pred_type_map, pred_type_maps, 'aji')
        avg_aji_value, list_aji_value, m_aji_list_process = calculate_m(mu_aji, 'aji')
        m_aji_list = elementwise_addition(m_aji_list, m_aji_list_process)

        mu_pq = m_ca(true_type_map, true_type_maps, pred_type_map, pred_type_maps, 'pq')
        avg_pq_value, list_pq_value, m_pq_list_process = calculate_m(mu_pq, 'pq')
        m_pq_list = elementwise_addition(m_pq_list, m_pq_list_process)

        fast_pq = pq_list(true, pred, match_iou=0.5)
        fast_aji_pre_list = aji_list(true, pred)
        dice_1_pre_list = get_dice_1_get_list(true, pred)
        fast_aji_plus_pre_list = aji_plus_list(true, pred)
        mpq_ll = mpq_list(true, true_type_map, pred, pred_type_map)

        # Accumulate metrics
        metrics[0].append(2 * calculate_ratio(dice_1_pre_list))
        metrics[1].append(2 * calculate_ratio(dice_1_list_process))
        metrics[2].append(calculate_ratio(fast_aji_pre_list))
        metrics[3].append(calculate_ratio(fast_aji_plus_pre_list))

        pq_result = calculate_pq(fast_pq)
        metrics[4].append(pq_result[0])
        metrics[5].append(pq_result[1].cpu().numpy())
        metrics[6].append(pq_result[2].cpu().numpy())

        pq_result = calculate_pq(mpq_ll)
        metrics[7].append(pq_result[0])
        metrics[8].append(pq_result[1].cpu().numpy())
        metrics[9].append(pq_result[2].cpu().numpy())

    # Calculate final metrics
    metrics_array = np.array(metrics)
    metrics_avg = np.mean(metrics_array, axis=-1)

    m_metrics = [[], [], [], [], []]
    m_metrics[0].append(2 * calculate_ratio(dice_1_list_process))
    m_metrics[1].append(calculate_ratio(m_aji_list))

    pq_result = calculate_pq(m_pq_list)
    m_metrics[2].append(pq_result[0])
    m_metrics[3].append(pq_result[1].cpu().numpy())
    m_metrics[4].append(pq_result[2].cpu().numpy())

    m_metrics_array = np.array(m_metrics)

    # Save results
    flat_metrics = metrics_avg.flatten()
    flat_m_metrics_array = m_metrics_array.flatten()

    with open(f'{os.path.dirname(pred_dir)}/metrics_miji.txt', 'a') as f:
        f.write("dice, dice1_list, aji, aji_plus, b_dq, b_sq, b_pq, m_dq_avg, m_sq_avg, m_pq_avg\n")
        np.savetxt(f, [flat_metrics], fmt='%.5f', delimiter='\t')
        f.write("m_dice, m_aji, m_dq_list, m_sq_list, m_pq_list\n")
        np.savetxt(f, [flat_m_metrics_array], fmt='%.5f', delimiter='\t')

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nuclear Segmentation Metrics Evaluation")

    parser.add_argument(
        "--mode",
        help="Evaluation mode: 'type' for type classification, 'instance' for instance segmentation",
        nargs="?",
        default="instance",
        const="instance",
        choices=["instance", "type"],
    )

    parser.add_argument(
        "--n_pred_dir",
        help="Directory containing prediction results",
        nargs="?",
        default="/path/to/predictions",
    )

    parser.add_argument(
        "--n_true_dir",
        help="Directory containing ground truth annotations",
        nargs="?",
        default="/path/to/ground_truth",
    )

    args = parser.parse_args()

    print(f"Running in {args.mode} mode")
    print(f"Prediction directory: {args.n_pred_dir}")
    print(f"Ground truth directory: {args.n_true_dir}")

    if args.mode == "instance":
        # Run instance segmentation evaluation
        run_nuclei_type_stat(args.n_pred_dir, args.n_true_dir)
        run_nuclei_fast_instance_(args.n_pred_dir, args.n_true_dir, print_img_stats=True)
    elif args.mode == "type":
        # Run type classification evaluation
        run_nuclei_type_stat(args.n_pred_dir, args.n_true_dir)