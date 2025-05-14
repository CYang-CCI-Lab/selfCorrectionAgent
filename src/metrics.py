import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.utils import resample
from scipy import stats

# new metrics
def t14_calculate_metrics(true_labels: pd.Series, predictions: pd.Series) -> dict:

    # Check for valid inputs
    if len(true_labels) != len(predictions):
        raise ValueError("The length of true_labels and predictions must be the same.")

    if any(not isinstance(pred, str) for pred in predictions):
        raise ValueError("All predictions must be non-null strings.")
  
    true_labels = true_labels.apply(lambda x: f"T{x+1}")

    metrics = {}
    label_counts = {}

    for label in set(true_labels):
        metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
        label_counts[label] = 0

    for true_label, prediction in zip(true_labels, predictions):
        prediction = prediction.upper()
        label_counts[true_label] += 1
        if true_label in prediction:
            metrics[true_label]["tp"] += 1
        else:
            metrics[true_label]["fn"] += 1

        for label in metrics:
            if label in prediction and label != true_label:
                metrics[label]["fp"] += 1

    results = {}
    total_tp = total_fp = total_fn = 0
    macro_precision = macro_recall = macro_f1 = 0
    total_instances = len(true_labels)

    for label, counts in metrics.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = label_counts[label]

        results[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
            "num_errors": fp + fn,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    # Calculate macro-averaged metrics
    num_labels = len(metrics)
    macro_precision /= num_labels
    macro_recall /= num_labels
    macro_f1 /= num_labels

    # Calculate overall (micro-averaged) precision, recall, and F1 score
    total_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0
    )

    # Calculate weighted (balanced) F1 score
    weighted_f1 = (
        sum(results[label]["f1"] * label_counts[label] for label in metrics)
        / total_instances
    )

    results["overall"] = {
        # 'precision': round(total_precision, 3),
        # 'recall': round(total_recall, 3),
        # 'f1': round(total_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
        # 'weighted_f1': round(weighted_f1, 3),
        "support": total_instances,
        "num_errors": total_fp + total_fn,
    }
    # print(f"Invalid predictions: {invalid_cnt}")
    return results


def n03_calculate_metrics(true_labels: pd.Series, predictions: pd.Series) -> dict:

    # Check for valid inputs
    if len(true_labels) != len(predictions):
        raise ValueError("The length of true_labels and predictions must be the same.")

    if any(not isinstance(pred, str) for pred in predictions):
        raise ValueError("All predictions must be non-null strings.")

    true_labels = true_labels.apply(lambda x: f"N{x}")

    metrics = {}
    label_counts = {}

    for label in set(true_labels):
        metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
        label_counts[label] = 0

    for true_label, prediction in zip(true_labels, predictions):
        prediction = prediction.upper()
        prediction = prediction.replace("NO", "N0").replace("NL", "N1")
        label_counts[true_label] += 1
        if true_label in prediction:
            metrics[true_label]["tp"] += 1
        else:
            metrics[true_label]["fn"] += 1

        for label in metrics:
            if label in prediction and label != true_label:
                metrics[label]["fp"] += 1

    results = {}
    total_tp = total_fp = total_fn = 0
    macro_precision = macro_recall = macro_f1 = 0
    total_instances = len(true_labels)

    for label, counts in metrics.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = label_counts[label]

        results[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
            "num_errors": fp + fn,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    # Calculate macro-averaged metrics
    num_labels = len(metrics)
    macro_precision /= num_labels
    macro_recall /= num_labels
    macro_f1 /= num_labels

    # Calculate overall (micro-averaged) precision, recall, and F1 score
    total_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    )
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = (
        2 * total_precision * total_recall / (total_precision + total_recall)
        if (total_precision + total_recall) > 0
        else 0
    )

    # Calculate weighted (balanced) F1 score
    weighted_f1 = (
        sum(results[label]["f1"] * label_counts[label] for label in metrics)
        / total_instances
    )

    results["overall"] = {
        # 'precision': round(total_precision, 3),
        # 'recall': round(total_recall, 3),
        # 'f1': round(total_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3),
        # 'weighted_f1': round(weighted_f1, 3),
        "support": total_instances,
        "num_errors": total_fp + total_fn,
    }

    return results



# def t14_calculate_metrics2(true_labels: pd.Series, predictions: pd.Series) -> dict:
#     """
#     Calculate classification metrics after removing invalid rows.
#     Invalid rows: rows where 'predictions' is not a string or is null,
#                   or where 'true_labels' is null (if that applies).
#     Returns per-label, macro, and micro-averaged metrics.
#     """

#     # 1) Create a valid mask
#     valid_mask = (
#         true_labels.notnull() &
#         predictions.notnull() &
#         predictions.apply(lambda x: isinstance(x, str))
#     )

#     # 2) Filter data using the valid_mask
#     filtered_true_labels = true_labels[valid_mask]
#     filtered_predictions = predictions[valid_mask]

#     # 3) Convert true labels to "T1", "T2", ...
#     filtered_true_labels = filtered_true_labels.apply(lambda x: f"T{x+1}")

#     # 4) Convert predictions to uppercase
#     filtered_predictions = filtered_predictions.apply(lambda x: x.upper())

#     # Prepare data structures
#     metrics = {}
#     label_counts = {}

#     # Identify all possible labels in the filtered data
#     unique_labels = set(filtered_true_labels)
#     for label in unique_labels:
#         metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
#         label_counts[label] = 0

#     # Populate tp, fp, fn
#     for true_label, prediction in zip(filtered_true_labels, filtered_predictions):
#         label_counts[true_label] += 1

#         if true_label in prediction:
#             metrics[true_label]["tp"] += 1
#         else:
#             metrics[true_label]["fn"] += 1

#         # For all labels that appear in the prediction (besides the true one), mark fp
#         for label in unique_labels:
#             if label in prediction and label != true_label:
#                 metrics[label]["fp"] += 1

#     # Compute metrics
#     results = {}
#     total_tp = total_fp = total_fn = 0
#     macro_precision = macro_recall = macro_f1 = 0
#     total_instances = len(filtered_true_labels)

#     # Per-label metrics
#     for label, counts in metrics.items():
#         tp = counts["tp"]
#         fp = counts["fp"]
#         fn = counts["fn"]

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
#         support   = label_counts[label]

#         results[label] = {
#             "precision": round(precision, 3),
#             "recall":    round(recall, 3),
#             "f1":        round(f1, 3),
#             "support":   support,
#             "num_errors": fp + fn,
#         }

#         total_tp += tp
#         total_fp += fp
#         total_fn += fn

#         macro_precision += precision
#         macro_recall += recall
#         macro_f1 += f1

#     # Macro-averaged metrics
#     num_labels = len(unique_labels)
#     if num_labels > 0:
#         macro_precision /= num_labels
#         macro_recall /= num_labels
#         macro_f1 /= num_labels

#     # --- Micro-averaged metrics ---
#     # total_tp, total_fp, and total_fn are the sums across all labels
#     micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
#     micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
#     micro_f1        = (
#         2 * micro_precision * micro_recall / (micro_precision + micro_recall)
#         if (micro_precision + micro_recall) > 0
#         else 0
#     )

#     # Weighted F1 (commented out, but you can keep if needed)
#     weighted_f1 = (
#         sum(results[label]["f1"] * label_counts[label] for label in metrics) / total_instances
#         if total_instances > 0
#         else 0
#     )

#     # Add overall metrics
#     results["overall"] = {
#         "macro_precision": round(macro_precision, 3),
#         "macro_recall":    round(macro_recall, 3),
#         "macro_f1":        round(macro_f1, 3),
#         "micro_precision": round(micro_precision, 3),
#         "micro_recall":    round(micro_recall, 3),
#         "micro_f1":        round(micro_f1, 3),
#         "weighted_f1":      round(weighted_f1, 3),
#         "support":         total_instances,
#         "num_errors":      total_fp + total_fn,
#     }

#     # Number of invalid rows
#     total_rows = len(true_labels)
#     invalid_count = total_rows - total_instances
#     # print(f"Number of invalid rows removed: {invalid_count}")

#     return results


# def n03_calculate_metrics2(true_labels: pd.Series, predictions: pd.Series) -> dict:
#     """
#     Calculate classification metrics for N staging (N0, N1, N2, N3) after removing invalid rows.
#     Invalid rows: rows where 'predictions' is not a string or is null,
#                   or where 'true_labels' is null (if that applies).
#     Returns per-label, macro, and micro-averaged metrics.
#     """

#     # 1) Create a valid mask
#     valid_mask = (
#         true_labels.notnull() &
#         predictions.notnull() &
#         predictions.apply(lambda x: isinstance(x, str))
#     )

#     # 2) Filter data using the valid_mask
#     filtered_true_labels = true_labels[valid_mask]
#     filtered_predictions = predictions[valid_mask]

#     # 3) Convert true labels to "N0", "N1", ...
#     filtered_true_labels = filtered_true_labels.apply(lambda x: f"N{x}")

#     # 4) Convert predictions to uppercase (e.g. 'n2' -> 'N2')
#     filtered_predictions = filtered_predictions.apply(lambda x: x.upper())

#     # Prepare data structures
#     metrics = {}
#     label_counts = {}

#     # Identify all possible labels in the filtered data (e.g., {"N0", "N1", "N2", "N3"})
#     unique_labels = set(filtered_true_labels)
#     for label in unique_labels:
#         metrics[label] = {"tp": 0, "fp": 0, "fn": 0}
#         label_counts[label] = 0

#     # Populate tp, fp, fn
#     for true_label, prediction in zip(filtered_true_labels, filtered_predictions):
#         label_counts[true_label] += 1

#         # If the prediction matches the true label, increment tp; otherwise, fn for that label
#         if true_label in prediction:
#             metrics[true_label]["tp"] += 1
#         else:
#             metrics[true_label]["fn"] += 1

#         # For all labels that appear in the prediction (besides the true one), mark fp
#         for label in unique_labels:
#             if label in prediction and label != true_label:
#                 metrics[label]["fp"] += 1

#     # Compute metrics
#     results = {}
#     total_tp = total_fp = total_fn = 0
#     macro_precision = macro_recall = macro_f1 = 0
#     total_instances = len(filtered_true_labels)

#     # Per-label metrics
#     for label, counts in metrics.items():
#         tp = counts["tp"]
#         fp = counts["fp"]
#         fn = counts["fn"]

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
#         f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
#         support   = label_counts[label]

#         results[label] = {
#             "precision": round(precision, 3),
#             "recall":    round(recall, 3),
#             "f1":        round(f1, 3),
#             "support":   support,
#             "num_errors": fp + fn,
#         }

#         total_tp += tp
#         total_fp += fp
#         total_fn += fn

#         macro_precision += precision
#         macro_recall += recall
#         macro_f1 += f1

#     # Macro-averaged metrics
#     num_labels = len(unique_labels)
#     if num_labels > 0:
#         macro_precision /= num_labels
#         macro_recall /= num_labels
#         macro_f1 /= num_labels

#     # Micro-averaged metrics
#     micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
#     micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
#     micro_f1        = (
#         2 * micro_precision * micro_recall / (micro_precision + micro_recall)
#         if (micro_precision + micro_recall) > 0
#         else 0
#     )

#     # Weighted F1 (optional)
#     weighted_f1 = (
#         sum(results[label]["f1"] * label_counts[label] for label in metrics) / total_instances
#         if total_instances > 0
#         else 0
#     )

#     # Add overall metrics
#     results["overall"] = {
#         "macro_precision": round(macro_precision, 3),
#         "macro_recall":    round(macro_recall, 3),
#         "macro_f1":        round(macro_f1, 3),
#         "micro_precision": round(micro_precision, 3),
#         "micro_recall":    round(micro_recall, 3),
#         "micro_f1":        round(micro_f1, 3),
#         "weighted_f1":     round(weighted_f1, 3),
#         "support":         total_instances,
#         "num_errors":      total_fp + total_fn,
#     }

#     # For debugging: how many rows were invalid
#     total_rows = len(true_labels)
#     invalid_count = total_rows - total_instances
#     # print(f"Number of invalid rows removed: {invalid_count}")

#     return results