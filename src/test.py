from agent import ConditionalMemoryAgent
from prompt import system_instruction
from sklearn.model_selection import KFold
from prompt import baseline_prompt_n03 as baseline_prompt
from prompt import initial_predict_prompt_n03 as initial_predict_prompt
from prompt import subsequent_predict_prompt_n03 as subsequent_predict_prompt
from prompt import testing_predict_prompt_n03 as testing_predict_prompt

from openai import OpenAI
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Union


class TrainingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage") 
    rules: List[str] = Field(description="list of rules") 

class TestingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    # reasoning: str = Field(description="reasoning to support predicted cancer stage") 

class FixedTestSizeCV:
    def __init__(self, num_test_points=80, num_folds=5):
        self.num_test_points = num_test_points
        self.num_folds = num_folds

    def split(self, X, y=None):
        kf = KFold(n_splits=self.num_folds)
        indices = np.arange(len(X))
        
        for train_index, test_index in kf.split(indices):
            if len(test_index) > self.num_test_points:
                np.random.shuffle(test_index)
                test_index = test_index[:self.num_test_points]

            yield train_index, test_index
   
if __name__ == "__main__":
    openai_api_key = "Empty"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
    brca_report = brca_report[brca_report["n"]!=-1]
    sorted_df = brca_report.reset_index(drop=True)

    training_schema = TrainingResponse.model_json_schema()
    testing_schema = TestingResponse.model_json_schema()

    # for size in range(90, 101, 5):
    cv = FixedTestSizeCV()
    for fold, (test_idx, train_idx) in enumerate(cv.split(sorted_df)):
        memory_agent = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt,
                                                "learning_prompt":subsequent_predict_prompt,
                                                "testing_prompt":testing_predict_prompt},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "n")
        df_train, df_test = sorted_df.iloc[train_idx], sorted_df.iloc[test_idx]

        train_result = memory_agent.train(df_train)
        train_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03/{fold}_fold_saturation_train_result_80.csv", index=False)
        
        test_result = memory_agent.test(df_test)
        test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03/{fold}_fold_saturation_test_result_80.csv", index=False)

