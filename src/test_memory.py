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

    # t14
    memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
                                                "learning_prompt":subsequent_predict_prompt_t14,
                                                "testing_prompt":testing_predict_prompt_t14},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "t")
    t_train_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_1.csv")[["patient_filename","text","t"]]
    t_train_result = memory_agent_t14.train(t_train_data, 1)
    t_train_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t_train_result_1_newPrompt.csv")

    # test
    t_intersect_df = pd.read_csv("/home/yl3427/cylab/selfCorrectionAgent/src/t_intersect_for_newPrompt.csv")
    t_intersect_test_result = memory_agent_t14.test(t_intersect_df, 1)
    t_intersect_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/t_intersect_test_result_1_newPrompt.csv")
    
    t_only_memory_df = pd.read_csv("/home/yl3427/cylab/selfCorrectionAgent/src/t_only_memory_for_newPrompt.csv")
    t_only_memory_test_result = memory_agent_t14.test(t_only_memory_df, 1)
    t_only_memory_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/t_only_memory_test_result_1_newPrompt.csv")

    t_only_zscot_df = pd.read_csv("/home/yl3427/cylab/selfCorrectionAgent/src/t_only_zscot_for_newPrompt.csv")
    t_only_zscot_test_result = memory_agent_t14.test(t_only_zscot_df, 1)
    t_only_zscot_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/t_only_zscot_test_result_1_newPrompt.csv")

    
    # N03
    memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
                                                "learning_prompt":subsequent_predict_prompt_n03,
                                                "testing_prompt":testing_predict_prompt_n03},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "n")
    

    n_train_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_rain_1.csv")[["patient_filename","text","n"]]
    n_train_result = memory_agent_n03.train(n_train_data, 1)
    n_train_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n_train_result_1_newPrompt.csv")

    # test
    n_intersect_df = pd.read_csv("/home/yl3427/cylab/selfCorrectionAgent/src/n_intersect_for_newPrompt.csv")
    n_intersect_test_result = memory_agent_n03.test(n_intersect_df, 1)
    n_intersect_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/n_intersect_test_result_1_newPrompt.csv")
    
    n_only_memory_df = pd.read_csv("/home/yl3427/cylab/selfCorrectionAgent/src/n_only_memory_for_newPrompt.csv")
    n_only_memory_test_result = memory_agent_n03.test(n_only_memory_df, 1)
    n_only_memory_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/n_only_memory_test_result_1_newPrompt.csv")

    n_only_zscot_df = pd.read_csv("/home/yl3427/cylab/selfCorrectionAgent/src/n_only_zscot_for_newPrompt.csv")
    n_only_zscot_test_result = memory_agent_n03.test(n_only_zscot_df, 1)
    n_only_zscot_test_result.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/n_only_zscot_test_result_1_newPrompt.csv")

    