from agent import ConditionalMemoryAgent
from prompt import *
from metrics import *
from openai import OpenAI
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Union
from collections import defaultdict

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

    for i in range(2, 10):
        # t14
        memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
                                                    "learning_prompt":subsequent_predict_prompt_t14,
                                                    "testing_prompt":testing_predict_prompt_t14},
                            schema_dict={"learning_schema":training_schema,
                                            "testing_schema":testing_schema},
                                            label = "t")
        t_train_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_{i}.csv")[["patient_filename","text","t"]]
        t_train_result = memory_agent_t14.train(t_train_data, i)
        t_train_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_{i}_newPrompt.csv")

        # test
        t_test_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{i}.csv")[["patient_filename","text","t"]]
        t_test_result = memory_agent_t14.test(t_test_data, i)
        t_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{i}_newPrompt.csv")
        

        
        # N03
        memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                            prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
                                                    "learning_prompt":subsequent_predict_prompt_n03,
                                                    "testing_prompt":testing_predict_prompt_n03},
                            schema_dict={"learning_schema":training_schema,
                                            "testing_schema":testing_schema},
                                            label = "n")
        

        n_train_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_{i}.csv")[["patient_filename","text","n"]]
        n_train_result = memory_agent_n03.train(n_train_data, i)
        n_train_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_{i}_newPrompt.csv")

        # test
        n_test_data = pd.read_csv("/home/yl3427/cylab/selfCorrectionAgent/result/n03_test_{i}.csv")[["patient_filename","text","n"]]
        n_test_result = memory_agent_n03.test(n_test_data, i)
        n_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/n03_test_{i}_newPrompt.csv")

        