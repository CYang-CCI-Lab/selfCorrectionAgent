from agent import ConditionalMemoryAgent
from prompt import *
from openai import OpenAI
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Union
from datetime import datetime

class TrainingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage") 
    rules: List[str] = Field(description="list of rules") 

class TestingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage") 


if __name__ == "__main__":
    client = OpenAI(
        api_key = "empty",
        base_url = "http://localhost:8000/v1"
    )

    training_schema = TrainingResponse.model_json_schema()
    testing_schema = TestingResponse.model_json_schema()


    for run in range(10):

        # training t14 data to extract memory
        train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_{run}.csv"
        train_data = pd.read_csv(train_file_path)[['patient_filename', 'text', 't', 'n']]

        memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
                                                    "learning_prompt":subsequent_predict_prompt_t14,
                                                    "testing_prompt":testing_predict_prompt_t14},
                            schema_dict={"learning_schema":training_schema,
                                            "testing_schema":testing_schema},
                                            label = "t")
    
        train_result = memory_agent_t14.train(train_data)
        train_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_memory_dataset{run}.csv", index=False)


        # training n03 data to extract memory
        train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_{run}.csv"
        train_data = pd.read_csv(train_file_path)[['patient_filename', 'text', 't', 'n']]

        memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
                                                    "learning_prompt":subsequent_predict_prompt_n03,
                                                    "testing_prompt":testing_predict_prompt_n03},
                            schema_dict={"learning_schema":training_schema,
                                            "testing_schema":testing_schema},
                                            label = "n")
        
        train_result = memory_agent_n03.train(train_data)
        train_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_memory_dataset{run}.csv", index=False)
        
