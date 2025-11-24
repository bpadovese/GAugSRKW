#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py

Evaluation utilities for segment-based acoustic detection and classification tasks.
Implements precision, recall, F1-score, and PR-curve computation using
interval-overlap logic between reference annotations and predicted detections.

Author
------
Bruno Padovese (HALLO Project, SFU)
https://github.com/bpadovese
"""

# =============================================================================
# Imports
# =============================================================================

from pathlib import Path
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from intervaltree import Interval, IntervalTree

# =============================================================================
# Helper Functions
# =============================================================================

def build_interval_tree(df, start_col='start', end_col='end'):
    """
    Build an IntervalTree from a DataFrame of start and end times.

    This function converts a DataFrame of time intervals (acoustic detections
    or reference annotations in this case) into an IntervalTree, which supports efficient
    lookup of overlapping time ranges.

    The resulting tree can answer queries such as whether any interval overlaps
    a given time segment (via `tree.overlaps(start, end)`) or return all overlapping
    intervals (via `tree.search(start, end)`), e.g..
     
    "Which detections overlap the segment between 32.5 and 33.8 seconds?"

    Without an interval tree, such queries would require iterating through every row
    in the dataset, which is O(n). IntervalTree allows O(log n) queries after an 
    initial O(n log n) construction time.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing start and end columns.
    start_col : str, optional
        Column name representing interval start times. Default is 'start'.
    end_col : str, optional
        Column name representing interval end times. Default is 'end'.

    Returns
    -------
    tree : IntervalTree
        IntervalTree object for fast overlap queries.
    """
    tree = IntervalTree()
    for row in df.itertuples(index=False):
        start = getattr(row, start_col)
        end = getattr(row, end_col)
        if start < end:  # adding valid intervals to the tree
            tree.add(Interval(start, end))
        else:
            continue  
    return tree

def prebuild_reference_trees(reference_df, start_col='start', end_col='end'):
    """
    Group reference annotations by filename and precompute IntervalTrees.

    This function organizes ground-truth annotations (reference) into 
    separate IntervalTrees, one per filename. 
     
    This enables fast overlap checks between predicted and reference intervals 
    within each file.

    After execution, intervals and their corresponding trees can be accessed as:
        ref_by_file['myfile.wav']      # DataFrame of annotations for this file
        ref_trees['myfile.wav']        # IntervalTree for this file


    Parameters
    ----------
    reference_df : pandas.DataFrame
        Ground truth annotations with 'filename', 'start', and 'end' columns.
    start_col : str, optional
        Start column name. Default is 'start'.
    end_col : str, optional
        End column name. Default is 'end'.

    Returns
    -------
    ref_by_file : dict
        Mapping of filename -> reference DataFrame subset.
    ref_trees : dict
        Mapping of filename -> IntervalTree built from the reference intervals.
    """

    # Here each file's annotations dataframe is stored in a dictionary keyed by filename
    ref_by_file = dict(tuple(reference_df.groupby('filename')))

    # Here we build an interval tree for each file's annotations also keyed by filename
    ref_trees = {fn: build_interval_tree(df, start_col, end_col) for fn, df in ref_by_file.items()}

    return ref_by_file, ref_trees

def calculate_metrics(TP, FP, FN, total_time_units=None):
    """
    Compute classification metrics from confusion matrix components.

    Parameters
    ----------
    TP : int
        True positives.
    FP : int
        False positives.
    FN : int
        False negatives.
    total_time_units : float, optional
        Total duration (e.g., hours or minutes) to normalize FP rate.

    Returns
    -------
    metrics : dict
        Dictionary containing Precision, Recall, F1-Score, and optionally
        FPR per time unit.

    """

    metrics = {}

    if TP + FP == 0:
        metrics['Precision'] = 0
    else:
        metrics['Precision'] = TP / (TP + FP)

    if TP + FN == 0:
        metrics['Recall'] = 0
    else:
        metrics['Recall'] = TP / (TP + FN)

    if metrics['Precision'] + metrics['Recall'] == 0:
        metrics['F1-Score'] = 0
    else:
        metrics['F1-Score'] = 2 * metrics['Precision'] * metrics['Recall'] / (metrics['Precision'] + metrics['Recall'])

    # Calculate FPR per hour if total_time_units is provided
    if total_time_units is not None:
        if total_time_units == 0:
            metrics['FPR_per_time_unit'] = 0
        else:
            metrics['FPR_per_time_unit'] = FP / total_time_units

    return metrics


# =============================================================================
# Core Evaluation Functions
# =============================================================================

def get_continuous_results(eval_df, ref_df,
                           ref_by_file=None, ref_trees=None):
    """
    Compute TP, FP, and FN based on temporal overlaps between detections and references.

    Parameters
    ----------
    eval_df : pandas.DataFrame
        Evaluation results (predicted detections) with 'filename', 'start', 'end'.
    ref_df : pandas.DataFrame
        Ground truth reference annotations.
    ref_by_file : dict, optional
        Precomputed mapping of filename -> reference subset.
    ref_trees : dict, optional
        Precomputed mapping of filename -> reference IntervalTree.

    Returns
    -------
    dict
        {'TP': int, 'FP': int, 'FN': int}
    """
    tp = fp = fn = 0

    # -------------------------------------------------------------------------
    # If reference trees or file groups were not precomputed, build them now.
    # Each file gets its own DataFrame (ref_by_file) and corresponding IntervalTree (ref_trees)
    # For more information see the specific function docstrings.
    # -------------------------------------------------------------------------
    if ref_by_file is None or ref_trees is None:
        ref_by_file = dict(tuple(ref_df.groupby('filename')))
        ref_trees = {fn: build_interval_tree(df) for fn, df in ref_by_file.items()}

    # -------------------------------------------------------------------------
    # Similarly, group the evaluation (predicted detections) by filename
    # and build one IntervalTree per file.
    # -------------------------------------------------------------------------
    eval_by_file = dict(tuple(eval_df.groupby('filename')))
    eval_trees = {fn: build_interval_tree(df) for fn, df in eval_by_file.items()}

    # True positives / false negatives
    for filename, ref_tree in ref_trees.items():
        # Get the corresponding prediction tree for this file 
        eval_tree = eval_trees.get(filename, IntervalTree())
        
        for row in ref_by_file[filename].itertuples(index=False):
            if eval_tree.overlaps(row.start, row.end):
                tp += 1
            else:
                fn += 1

    # False positives
    for filename, eval_tree in eval_trees.items():
        # Get the reference tree for this file
        ref_tree = ref_trees.get(filename, IntervalTree())
        
        for row in eval_by_file[filename].itertuples(index=False):
            if not ref_tree.overlaps(row.start, row.end):
                fp += 1

    return {'TP': tp, 'FP': fp, 'FN': fn}

def evaluate_thresholded(evaluation_path, reference_path,
                         threshold_min=0, threshold_max=1, threshold_inc=0.05,
                         total_time_units=None, output_folder=None):
    """
    Evaluate detection results across a range of thresholds.

    Parameters
    ----------
    evaluation_path : str
        Path to CSV file containing predicted detections.
    reference_path : str
        Path to CSV file containing ground truth annotations.
    threshold_min, threshold_max, threshold_inc : float, optional
        Range and increment for threshold sweep.
    total_time_units : float, optional
        Total duration for normalization of FP per unit time.
    output_folder : str, optional
        Folder to save resulting metrics CSV.

    Returns
    -------
    metrics_df : pandas.DataFrame
        DataFrame containing computed metrics for each threshold.
    """
    eval_df = pd.read_csv(evaluation_path)
    ref_df = pd.read_csv(reference_path)
    thresholds = np.arange(threshold_min, threshold_max, threshold_inc)

    results = []
    for thr in tqdm(thresholds, desc="Evaluating thresholds"):
        filtered = eval_df[eval_df['score'] >= thr]
        result = get_continuous_results(filtered, ref_df)
        metrics = calculate_metrics(**result, total_time_units=total_time_units)
        metrics['Threshold'] = round(thr, 5)
        results.append(metrics)

    metrics_df = pd.DataFrame(results)

    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path / 'metrics.csv', index=False)
        print(f"Metrics saved to {output_path / 'metrics.csv'}")

    return metrics_df

def compute_segment_pr_curve(eval_df, ref_df, thresholds):
    """
    Compute a segment-aware Precision-Recall curve.

    This function evaluates detection performance at the *segment level*,
    rather than per frame or sample. Each detection and ground-truth event
    is represented as a time interval (start-end). A detection counts as a
    True Positive if it temporally overlaps any reference interval, and as a
    False Positive otherwise. Recall is computed analogously for missed
    reference segments (False Negatives).

    The PR curve is built by sweeping over detection score thresholds and
    recomputing these overlap-based metrics at each level.

    Parameters
    ----------
    eval_df : pandas.DataFrame
        Evaluation DataFrame containing 'score', 'filename', 'start', and 'end'.
    ref_df : pandas.DataFrame
        Ground truth reference annotations.
    thresholds : list[float]
        Thresholds over which to compute the PR curve.

    Returns
    -------
    precision_list : list[float]
    recall_list : list[float]
    f1_list : list[float]
    thresholds : list[float]
    """
    precision_list = []
    recall_list = []
    f1_list = []

    # Pre-sort evaluation by descending score
    eval_df = eval_df.sort_values(by='score', ascending=False).reset_index(drop=True)

    ref_by_file, ref_trees = prebuild_reference_trees(ref_df)

    for threshold in tqdm(thresholds, desc="Computing PR curve"):
        filtered_eval = eval_df[eval_df['score'] >= threshold]
        results = get_continuous_results(filtered_eval, ref_df, ref_by_file, ref_trees)

        tp, fp, fn = results['TP'], results['FP'], results['FN']
        metrics = calculate_metrics(tp, fp, fn)
        precision_list.append(metrics['Precision'])
        recall_list.append(metrics['Recall'])
        f1_list.append(metrics['F1-Score'])

    # Filter out garbage points with very low recall
    filtered = [(p, r, f, t) for p, r, f, t in zip(precision_list, recall_list, f1_list, thresholds) if r >= 0.01]
    if not filtered:
        return [], [], [], []

    precision_list, recall_list, f1_list, thresholds = zip(*filtered)
    return list(precision_list), list(recall_list), list(f1_list), list(thresholds)

def evaluate_segment_pr_curve(evaluation_path, reference_path, output_folder=None):
    """
    Evaluate model detections by computing a segment-aware PR curve.

    This wrapper loads detection and reference CSV files, computes precision,
    recall, and F1 scores across all score thresholds, and optionally saves
    the results to a CSV file. It is equivalent to the score-based mode of
    evaluation, where detections are assessed at the segment (interval) level.

    Parameters
    ----------
    evaluation_path : str
        Path to the CSV file containing model detections with a 'score' column.
    reference_path : str
        Path to the CSV file containing ground-truth annotations.
    output_folder : str, optional
        Folder where to save the resulting PR curve CSV file. If None, results
        are not saved.

    Returns
    -------
    tuple
        (precision_list, recall_list, f1_list, thresholds)
    """
    eval_df = pd.read_csv(evaluation_path)
    ref_df = pd.read_csv(reference_path)
    thresholds = sorted(eval_df['score'].unique(), reverse=True)

    precision, recall, f1, thresholds = compute_segment_pr_curve(eval_df, ref_df, thresholds)

    if output_folder:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        df_curve = pd.DataFrame({
            'threshold': thresholds,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        df_curve.to_csv(output_path / "pr_curve.csv", index=False)
        print(f"Saved PR curve data to {output_path / 'pr_curve.csv'}")

    return precision, recall, f1, thresholds
        
def main():        
    parser = argparse.ArgumentParser(description="Evaluate acoustic detection performance.")
    parser.add_argument('reference', type=str, help='Path to ground truth CSV file.')
    parser.add_argument('--evaluation', type=str, required=True, help='Path to model evaluation CSV file.')
    parser.add_argument('--mode', choices=['thresholded', 'score_based'], default='thresholded',
                        help='Evaluation mode to run.')
    parser.add_argument('--threshold_min', type=float, default=0)
    parser.add_argument('--threshold_max', type=float, default=1)
    parser.add_argument('--threshold_inc', type=float, default=0.05)
    parser.add_argument('--total_time_units', type=float, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    
    args = parser.parse_args()

    if args.mode == 'thresholded':
        evaluate_thresholded(
            evaluation_path=args.evaluation,
            reference_path=args.reference,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_inc=args.threshold_inc,
            total_time_units=args.total_time_units,
            output_folder=args.output_folder,
        )
    elif args.mode == 'score_based':
        evaluate_segment_pr_curve(
            evaluation_path=args.evaluation,
            reference_path=args.reference,
            output_folder=args.output_folder
        )
    else:
        raise ValueError("Invalid mode. Choose either 'thresholded' or 'score_based'.")

if __name__ == "__main__":
    main()
