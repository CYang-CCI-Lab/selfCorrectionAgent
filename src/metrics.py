import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils import resample
from scipy import stats

# old metrics
def t14_performance_report(df, ans_col="ans_str"):
    df['Has_Valid_Prediction'] = df[ans_col].str.contains('T1|T2|T3|T4', case=False)
    coded_pred_list = []
    for _, row in df.iterrows():
        if "T1" in str(row[ans_col]):
            coded_pred_list.append(0)
        elif "T2" in str(row[ans_col]):
            coded_pred_list.append(1)
        elif "T3" in str(row[ans_col]):
            coded_pred_list.append(2)
        elif "T4" in str(row[ans_col]):
            coded_pred_list.append(3)
        else:
            # unvalid answers 
            # Has_Valid_Prediction == False
            coded_pred_list.append(-1)
    df['coded_pred'] = coded_pred_list

    effective_index = df["Has_Valid_Prediction"] == True
    coded_pred = df[effective_index]['coded_pred'].to_list()
    t_labels = df[effective_index]["t"].to_list()

    target_names = ['T1', 'T2', 'T3', 'T4']
    print(classification_report(t_labels, coded_pred, target_names=target_names))
    precision, recall, f1, _ = precision_recall_fscore_support(t_labels, coded_pred, average='macro')
    return precision, recall, f1

def n03_performance_report(df, ans_col="ans_str"):
    df['Has_Valid_Prediction'] = df[ans_col].str.contains('N0|N1|N2|N3', case=False)
    coded_pred_list = []
    for _, row in df.iterrows():
        row[ans_col] = str(row[ans_col])
        if "N0" in row[ans_col]:
            coded_pred_list.append(0)
        elif "N1" in row[ans_col]:
            coded_pred_list.append(1)
        elif "N2" in row[ans_col]:
            coded_pred_list.append(2)
        elif "N3" in row[ans_col]:
            coded_pred_list.append(3)
        else:
            # unvalid answers 
            # Has_Valid_Prediction == False
            coded_pred_list.append(-1)
    df['coded_pred'] = coded_pred_list

    effective_index = df["Has_Valid_Prediction"] == True
    coded_pred = df[effective_index]['coded_pred'].to_list()
    n_labels = df[effective_index]["n"].to_list()

    target_names = ['N0', 'N1', 'N2', 'N3']
    print(classification_report(n_labels, coded_pred, target_names=target_names))
    precision, recall, f1, _ = precision_recall_fscore_support(n_labels, coded_pred, average='macro')
    return precision, recall, f1


# new metrics
def t14_calculate_metrics(true_labels: pd.Series, predictions: pd.Series) -> dict:

    # Check for valid inputs
    if len(true_labels) != len(predictions):
        raise ValueError("The length of true_labels and predictions must be the same.")
    
    if any(not isinstance(pred, str) for pred in predictions):
        raise ValueError("All predictions must be non-null strings.")
    
    true_labels = true_labels.apply(lambda x: f'T{x+1}')

    metrics = {}
    label_counts = {}
    
    for label in set(true_labels):
        metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0}
        label_counts[label] = 0

    for true_label, prediction in zip(true_labels, predictions):
        prediction = prediction.upper()
        label_counts[true_label] += 1
        if true_label in prediction:
            metrics[true_label]['tp'] += 1
        else:
            metrics[true_label]['fn'] += 1
        
        for label in metrics:
            if label in prediction and label != true_label:
                metrics[label]['fp'] += 1
    
    results = {}
    total_tp = total_fp = total_fn = 0
    macro_precision = macro_recall = macro_f1 = 0
    total_instances = len(true_labels)
    
    for label, counts in metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = label_counts[label]
        
        results[label] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'support': support,
            'num_errors': fp + fn
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
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    # Calculate weighted (balanced) F1 score
    weighted_f1 = sum(results[label]['f1'] * label_counts[label] for label in metrics) / total_instances

    results['overall'] = {
        # 'precision': round(total_precision, 3),
        # 'recall': round(total_recall, 3),
        # 'f1': round(total_f1, 3),
        'macro_precision': round(macro_precision, 3),
        'macro_recall': round(macro_recall, 3),
        'macro_f1': round(macro_f1, 3),
        # 'weighted_f1': round(weighted_f1, 3),
        'support': total_instances,
        'num_errors': total_fp + total_fn
    }
    
    return results

def n03_calculate_metrics(true_labels: pd.Series, predictions: pd.Series) -> dict:

    # Check for valid inputs
    if len(true_labels) != len(predictions):
        raise ValueError("The length of true_labels and predictions must be the same.")
    
    if any(not isinstance(pred, str) for pred in predictions):
        raise ValueError("All predictions must be non-null strings.")
    
    true_labels = true_labels.apply(lambda x: f'N{x}')

    metrics = {}
    label_counts = {}
    
    for label in set(true_labels):
        metrics[label] = {'tp': 0, 'fp': 0, 'fn': 0}
        label_counts[label] = 0

    for true_label, prediction in zip(true_labels, predictions):
        prediction = prediction.upper()
        prediction = prediction.replace("NO", "N0").replace("NL", "N1")
        label_counts[true_label] += 1
        if true_label in prediction:
            metrics[true_label]['tp'] += 1
        else:
            metrics[true_label]['fn'] += 1
        
        for label in metrics:
            if label in prediction and label != true_label:
                metrics[label]['fp'] += 1
    
    results = {}
    total_tp = total_fp = total_fn = 0
    macro_precision = macro_recall = macro_f1 = 0
    total_instances = len(true_labels)
    
    for label, counts in metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = label_counts[label]
        
        results[label] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'support': support,
            'num_errors': fp + fn
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
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

    # Calculate weighted (balanced) F1 score
    weighted_f1 = sum(results[label]['f1'] * label_counts[label] for label in metrics) / total_instances

    results['overall'] = {
        # 'precision': round(total_precision, 3),
        # 'recall': round(total_recall, 3),
        # 'f1': round(total_f1, 3),
        'macro_precision': round(macro_precision, 3),
        'macro_recall': round(macro_recall, 3),
        'macro_f1': round(macro_f1, 3),
        # 'weighted_f1': round(weighted_f1, 3),
        'support': total_instances,
        'num_errors': total_fp + total_fn
    }
    
    return results
