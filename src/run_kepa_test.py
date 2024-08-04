from agent import *
from prompt import *
from openai import OpenAI
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Union
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
if not load_dotenv(find_dotenv()):
    raise Exception("Failed to load .env file")
from datetime import datetime
from typing import List, Optional, Tuple

# from langsmith import traceable
# from langsmith.wrappers import wrap_openai


class TestingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage") 


if __name__ == "__main__":
    client = OpenAI(
        api_key = "empty",
        base_url = "http://localhost:8000/v1", 
        timeout=120.0
    )
    
    testing_schema = TestingResponse.model_json_schema()

    # ### langsmith test ###

    # # Update with your API URL if using a hosted instance of Langsmith.
    # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    # os.environ["LANGCHAIN_API_KEY"] = "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"  # Update with your API key
    # os.environ["LANGCHAIN_TRACING_V2"] = "true"
    # os.environ["LANGCHAIN_PROJECT"] = "TNM staging"  # Optional: "default" is used if not set

    # ########

    for run in range(10):
       
        # extract memory for t14
        t_train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_memory_dataset{run}.csv"
        t_train_data = pd.read_csv(t_train_file_path)

        t_memory_tup = []
        for idx, row in t_train_data.iterrows():
            t_memory_tup.append((idx+1,row['cmem_t_memory_str']))
        t_memory_tup = t_memory_tup[9::10]

        # kepa test for t14
        t_test_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{run}.csv" # 700 reports
        t_test_data = pd.read_csv(t_test_file_path)[['patient_filename', 't', 'text']]

        kepa_test_agent_t14 = KEPATestAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1")

        t_test_result = kepa_test_agent_t14.dynamic_test(testing_dataset=t_test_data, prompt=testing_predict_prompt_t14, schema=testing_schema, label= 't', memory_tup= t_memory_tup)
        t_test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/0718_t14_dynamic_test_{run}_outof_10runs.csv", index=False)



        # extract memory for n03
        n_train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_memory_dataset{run}.csv"
        n_train_data = pd.read_csv(n_train_file_path)

        n_memory_tup = []
        for idx, row in n_train_data.iterrows():
            n_memory_tup.append((idx+1,row['cmem_n_memory_str']))
        n_memory_tup = n_memory_tup[9::10]

        # kepa test for n03
        n_test_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_test_{run}.csv"
        n_test_data = pd.read_csv(n_test_file_path)[['patient_filename', 'n', 'text']]

        kepa_test_agent_n03 = KEPATestAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1")

        n_test_result = kepa_test_agent_n03.dynamic_test(testing_dataset=n_test_data, prompt=testing_predict_prompt_n03, schema=testing_schema, label= 'n', memory_tup= n_memory_tup)
        n_test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/0718_n03_dynamic_test_{run}_outof_10runs.csv", index=False)
        