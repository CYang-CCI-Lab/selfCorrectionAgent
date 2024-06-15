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

    test_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_0.csv"
    test_data = pd.read_csv(test_file_path)[['Unnamed: 0', 'patient_filename', 't', 'text', 'type', 'n']][:30]

    # t14
    train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_0.csv"
    train_data = pd.read_csv(train_file_path)

    memory_tup = []
    for idx, row in train_data.iterrows():
        if row["cmem_t_is_updated"] == True:
            memory_tup.append((idx,row['cmem_t_memory_str']))
    memory_tup = memory_tup[80::5]


    memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
                                                "learning_prompt":subsequent_predict_prompt_t14,
                                                "testing_prompt":testing_predict_prompt_t14},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "t")
 
    test_result = memory_agent_t14.dynatic_test(test_data, memory_tup)
    test_result.to_csv(f"t14_dynamic_test_0_with_reas.csv", index=False)

    wandb.finish()


    # N03
    train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_0.csv"
    train_data = pd.read_csv(train_file_path)

    memory_tup = []
    for idx, row in train_data.iterrows():
        if row["cmem_n_is_updated"] == True:
            memory_tup.append((idx,row['cmem_n_memory_str']))

    memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
                                                "learning_prompt":subsequent_predict_prompt_n03,
                                                "testing_prompt":testing_predict_prompt_n03},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "n")
    
    test_result = memory_agent_n03.dynatic_test(test_data, memory_tup)
    test_result.to_csv(f"n03_dynamic_test_0_with_reas.csv", index=False)

    wandb.finish()