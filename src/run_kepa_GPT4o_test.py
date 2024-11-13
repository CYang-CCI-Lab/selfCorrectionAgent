from agent import *
from prompt import *
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv, find_dotenv
if not load_dotenv(find_dotenv()):
    raise Exception("Failed to load .env file")

class TestingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage") 


if __name__ == "__main__":

    for run in [0, 1, 2, 3, 4, 5, 6, 8]:
        # extract memory for t14
        t_train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_memory_dataset{run}.csv"
        t_train_data = pd.read_csv(t_train_file_path)

        t_memory_dict = {}
        for idx, row in t_train_data.iterrows():
            t_memory_dict[idx+1] = row['cmem_t_memory_str']
        
        t_memory = t_memory_dict[40]

        # kepa test for t14
        t_test_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{run}.csv" # 700 reports
        t_test_data = pd.read_csv(t_test_file_path)[['patient_filename', 't', 'text']]

        t_agent = GPTAgent('t')

        t_test_result = t_agent.test(testing_dataset=t_test_data, memory=t_memory)
        t_test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/1112_t14_gpt_test_{run}_outof_8runs.csv", index=False)


    for run in [0, 1, 3, 4, 5, 6, 7, 9]:
        # extract memory for n03
        n_train_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_memory_dataset{run}.csv"
        n_train_data = pd.read_csv(n_train_file_path)

        n_memory_dict = {}
        for idx, row in n_train_data.iterrows():
            n_memory_dict[idx+1] = row['cmem_n_memory_str']
        
        n_memory = n_memory_dict[40]

        # kepa test for n03
        n_test_file_path =  f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_test_{run}.csv"
        n_test_data = pd.read_csv(n_test_file_path)[['patient_filename', 'n', 'text']]

        n_agent = GPTAgent('n')

        n_test_result = n_agent.test(testing_dataset=n_test_data, memory=n_memory)
        n_test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/1112_n03_gpt_test_{run}_outof_8runs.csv", index=False)
