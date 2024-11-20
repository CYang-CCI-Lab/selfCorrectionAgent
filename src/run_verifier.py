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
    evaluation: str = Field(description="evaluation of the reasoning")


if __name__ == "__main__":
    client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)

    testing_schema = TestingResponse.model_json_schema()

    t_KEwLTM_file_path = "/home/yl3427/cylab/selfCorrectionAgent/result/0718_t14_dynamic_test_0_outof_10runs.csv"
    t_KEwLTM_df = pd.read_csv(t_KEwLTM_file_path)

    t_zscot_file_path = (
        "/home/yl3427/cylab/selfCorrectionAgent/result/0716_t14_zscot_test_800.csv"
    )
    t_zscot_df = pd.read_csv(t_zscot_file_path)
    t_zscot_df = t_zscot_df[
        t_zscot_df.patient_filename.isin(t_KEwLTM_df.patient_filename)
    ]

    verifier = PostHocVerificationAgent(
        client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    t_KEwLTM_result = verifier.verify(
        testing_dataset=t_KEwLTM_df,
        reasoning_column="cmem_t_40reasoning",
        prompt=verification_prompt_t14,
        schema=testing_schema,
        label="t",
    )
    t_KEwLTM_result.to_csv(
        "/home/yl3427/cylab/selfCorrectionAgent/result/0720_t14_dynamic_test_0_outof_10runs_verification.csv",
        index=False,
    )

    t_zscot_result = verifier.verify(
        testing_dataset=t_zscot_df,
        reasoning_column="zs_t_reasoning",
        prompt=verification_prompt_t14,
        schema=testing_schema,
        label="t",
    )
    t_zscot_result.to_csv(
        "/home/yl3427/cylab/selfCorrectionAgent/result/0720_t14_zscot_test_800_0_verification.csv",
        index=False,
    )

    n_KEwLTM_file_path = "/home/yl3427/cylab/selfCorrectionAgent/result/0718_n03_dynamic_test_1_outof_10runs.csv"
    n_KEwLTM_df = pd.read_csv(n_KEwLTM_file_path)

    n_zscot_file_path = (
        "/home/yl3427/cylab/selfCorrectionAgent/result/0716_n03_zscot_test_800.csv"
    )
    n_zscot_df = pd.read_csv(n_zscot_file_path)
    n_zscot_df = n_zscot_df[
        n_zscot_df.patient_filename.isin(n_KEwLTM_df.patient_filename)
    ]

    n_KEwLTM_result = verifier.verify(
        testing_dataset=n_KEwLTM_df,
        reasoning_column="cmem_n_40reasoning",
        prompt=verification_prompt_n03,
        schema=testing_schema,
        label="n",
    )
    n_KEwLTM_result.to_csv(
        "/home/yl3427/cylab/selfCorrectionAgent/result/0720_n03_dynamic_test_1_outof_10runs_verification.csv",
        index=False,
    )

    n_zscot_result = verifier.verify(
        testing_dataset=n_zscot_df,
        reasoning_column="zs_n_reasoning",
        prompt=verification_prompt_n03,
        schema=testing_schema,
        label="n",
    )
    n_zscot_result.to_csv(
        "/home/yl3427/cylab/selfCorrectionAgent/result/0720_n03_zscot_test_800_1_verification.csv",
        index=False,
    )

    print("Done")
