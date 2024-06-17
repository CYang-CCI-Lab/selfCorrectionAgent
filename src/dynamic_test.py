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

class TestingResponseWithoutReasoning(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
 

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



    for run in range(10):

        # test data (common for both t14 and n03)
        test_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{run}.csv" # 700 reports
        test_data = pd.read_csv(test_file_path)[['Unnamed: 0', 'patient_filename', 't', 'text', 'n']]

        # t14 training data to extract memory
        train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_{run}.csv"
        train_data = pd.read_csv(train_file_path)

        memory_tup = []
        for idx, row in train_data.iterrows():
            # if row["cmem_t_is_updated"] == True:
            memory_tup.append((idx+1,row['cmem_t_memory_str']))
        memory_tup = memory_tup[9::10]


        # t14 dynamic testing
        memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
                                                    "learning_prompt":subsequent_predict_prompt_t14,
                                                    "testing_prompt":testing_predict_prompt_t14},
                            schema_dict={"learning_schema":training_schema,
                                            "testing_schema":testing_schema},
                                            label = "t")
    
        test_result = memory_agent_t14.dynamic_test(test_data, memory_tup)
        test_result.to_csv(f"t14_dynamic_test_{run}_outof_10runs.csv", index=False)


        # N03 training data to extract memory
        train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_{run}.csv"
        train_data = pd.read_csv(train_file_path)

        memory_tup = []
        for idx, row in train_data.iterrows():
            # if row["cmem_t_is_updated"] == True:
            memory_tup.append((idx+1,row['cmem_n_memory_str']))
        memory_tup = memory_tup[9::10]
        
        memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
                                                    "learning_prompt":subsequent_predict_prompt_n03,
                                                    "testing_prompt":testing_predict_prompt_n03},
                            schema_dict={"learning_schema":training_schema,
                                            "testing_schema":testing_schema},
                                            label = "n")
        
        test_result = memory_agent_n03.dynamic_test(test_data, memory_tup)
        test_result.to_csv(f"n03_dynamic_test_{run}_outof_10runs.csv", index=False)

