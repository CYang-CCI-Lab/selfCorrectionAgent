import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils import resample
from scipy import stats
    
def t14_performance_report(df, ans_col="ans_str"):
    df['Has_Valid_Prediction'] = df[ans_col].str.contains('T1|T2|T3|T4', case=False)
    coded_pred_list = []
    for _, row in df.iterrows():
        if "T1" in row[ans_col]:
            coded_pred_list.append(0)
        elif "T2" in row[ans_col]:
            coded_pred_list.append(1)
        elif "T3" in row[ans_col]:
            coded_pred_list.append(2)
        elif "T4" in row[ans_col]:
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
    # precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro')


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
