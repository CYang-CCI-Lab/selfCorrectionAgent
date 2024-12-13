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
from typing import List, Optional, Tuple, Literal


class Response_T(BaseModel):
    reasoning: str = Field(
        description="Step-by-step explanation of how you interpreted the report to determine the T stage."
    )
    stage: Literal["T1", "T2", "T3", "T4"] = Field(
        description="The T stage determined from the report. Stage must be one of 'T1', 'T2', 'T3' or 'T4.'"
    )


class Response_N(BaseModel):
    reasoning: str = Field(
        description="Step-by-step explanation of how you interpreted the report to determine the N stage."
    )
    stage: Literal["N0", "N1", "N2", "N3"] = Field(
        description="The N stage determined from the report. Stage must be one of 'N0', 'N1', 'N2' or 'N3.'"
    )


if __name__ == "__main__":
    client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)

    #### T14
    t_schema = Response_T.model_json_schema()

    for run in [0, 1, 2, 3, 4, 5, 6, 8]:

        # extract memory for t14
        t_train_file_path = (
            f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_memory_dataset{run}.csv"
        )
        t_train_data = pd.read_csv(t_train_file_path)

        t_memory_dict = {}
        for idx, row in t_train_data.iterrows():
            t_memory_dict[idx + 1] = row["cmem_t_memory_str"]
        t_memory = t_memory_dict[40]

        # kepa test for t14
        t_test_file_path = f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{run}.csv"  # 700 reports
        t_test_data = pd.read_csv(t_test_file_path)[["patient_filename", "t", "text"]]

        t_kewltm_agent = Agent(
            client=client,
            model="meta-llama/Llama-3.3-70B-Instruct",
            label="t",
            schema=t_schema,
            test_name="t14_kewltm",
        )

        t_kewltm_result = t_kewltm_agent.test(
            testing_dataset=t_test_data,
            prompt=ltm_t14,
            context=t_memory,
        )
        t_kewltm_result.to_csv(
            f"/home/yl3427/cylab/selfCorrectionAgent/result/1208_t14_llama3_kewltm_{run}_outof_10runs.csv",
            index=False,
        )
        t_kewltm_result.to_csv(
            f"/home/yl3427/cylab/selfCorrectionAgent/result_backup/1208_t14_llama3_kewltm_{run}_outof_10runs.csv",
            index=False,
        )

    #### N03
    n_schema = Response_N.model_json_schema()
    for run in [0, 1, 3, 4, 5, 6, 7, 9]:
        # extract memory for n03
        n_train_file_path = (
            f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_memory_dataset{run}.csv"
        )
        n_train_data = pd.read_csv(n_train_file_path)

        n_memory_dict = {}
        for idx, row in n_train_data.iterrows():
            n_memory_dict[idx + 1] = row["cmem_n_memory_str"]

        n_memory = n_memory_dict[40]

        # kepa test for n03
        n_test_file_path = (
            f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_test_{run}.csv"
        )
        n_test_data = pd.read_csv(n_test_file_path)[["patient_filename", "n", "text"]]

        n_kewltm_agent = Agent(
            client=client,
            model="meta-llama/Llama-3.3-70B-Instruct",
            label="n",
            schema=n_schema,
            test_name="n03_kewltm",
        )

        n_kewltm_result = n_kewltm_agent.test(
            testing_dataset=n_test_data,
            prompt=ltm_n03,
            context=n_memory,
        )
        n_kewltm_result.to_csv(
            f"/home/yl3427/cylab/selfCorrectionAgent/result/1208_n03_llama3_kewltm_{run}_outof_10runs.csv",
            index=False,
        )
        n_kewltm_result.to_csv(
            f"/home/yl3427/cylab/selfCorrectionAgent/result_backup/1208_n03_llama3_kewltm_{run}_outof_10runs.csv",
            index=False,
        )
