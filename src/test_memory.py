from agent import ConditionalMemoryAgent, MemoryAgent
from prompt import *
from openai import OpenAI
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Union
from datetime import datetime
import wandb

class TrainingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage") 
    rules: List[str] = Field(description="list of rules") 

class TestingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage") 

def split_reports(df, num_train=100):
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    train_index = indices[:num_train]
    test_index = indices[num_train:]
    return train_index, test_index


if __name__ == "__main__":
    client = OpenAI(
        api_key = "empty",
        base_url = "http://localhost:8000/v1"
    )

    training_schema = TrainingResponse.model_json_schema()
    testing_schema = TestingResponse.model_json_schema()

    brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
    brca_report = brca_report[brca_report["n"]!=-1]
    sorted_df = brca_report.reset_index(drop=True)

    # t14
    memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
                                                "learning_prompt":subsequent_predict_prompt_t14,
                                                "testing_prompt":testing_predict_prompt_t14},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "t")
    for i in range(10):
        train_index, test_index = split_reports(sorted_df)
        df_training_samples = sorted_df.iloc[train_index]
        df_testing_samples = sorted_df.iloc[test_index]
  
        train_result = memory_agent_t14.train(df_training_samples, i)
        train_result.to_csv(f"t14_train_rules_{i}.csv", index=False)

        # test_result = memory_agent_t14.test(df_testing_samples, i)
        # test_result.to_csv(f"t14_test_rules_{i}.csv", index=False)

        memory_agent_t14.clear_memory()
    wandb.finish()

    # N03
    memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
                                                "learning_prompt":subsequent_predict_prompt_n03,
                                                "testing_prompt":testing_predict_prompt_n03},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "n")
    
    for i in range(10):
        train_index, test_index = split_reports(sorted_df)
        df_training_samples = sorted_df.iloc[train_index]
        df_testing_samples = sorted_df.iloc[test_index]
  
        train_result = memory_agent_n03.train(df_training_samples, i)
        train_result.to_csv(f"n03_train_rules_{i}.csv", index=False)

        # test_result = memory_agent_n03.test(df_testing_samples, i)
        # test_result.to_csv(f"n03_test_rules_{i}.csv", index=False)

        memory_agent_n03.clear_memory()
    wandb.finish()