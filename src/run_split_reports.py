import numpy as np
import pandas as pd

def split_reports(df, num_train=100):
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    train_index = indices[:num_train]
    test_index = indices[num_train:]
    return train_index, test_index

if __name__ == "__main__":

    brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
    brca_report = brca_report[brca_report["n"]!=-1]
    sorted_df = brca_report.reset_index(drop=True)

    for i in range(10):

        # T14
        t_train_index, t_test_index = split_reports(sorted_df)

        t_df_training_samples = sorted_df.iloc[t_train_index].drop(columns=["n"])
        t_df_training_samples.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_{i}.csv", index=False)

        t_df_testing_samples = sorted_df.iloc[t_test_index].drop(columns=["n"])
        t_df_testing_samples.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{i}.csv", index=False)  


        # N03
        n_train_index, n_test_index = split_reports(sorted_df)

        n_df_training_samples = sorted_df.iloc[n_train_index].drop(columns=["t"])
        n_df_training_samples.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_{i}.csv", index=False)

        n_df_testing_samples = sorted_df.iloc[n_test_index].drop(columns=["t"])
        n_df_testing_samples.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_test_{i}.csv", index=False)
    