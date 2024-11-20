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


class TestingResponseWithoutReasoning(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")


if __name__ == "__main__":
    client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)

    zs_testing_schema = TestingResponseWithoutReasoning.model_json_schema()

    brca_report = pd.read_csv(
        "/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv"
    )
    brca_report = brca_report[brca_report["n"] != -1]
    test_data = brca_report[["patient_filename", "t", "text", "n"]]

    zs_agent = ZSAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1")

    # zs test for t14
    t_zs_test_result = zs_agent.test(
        testing_dataset=test_data,
        prompt=zs_predict_prompt_t14,
        schema=zs_testing_schema,
        label="t",
    )
    t_zs_test_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result/0716_t14_zs_test_800.csv",
        index=False,
    )

    # zs test for n03
    n_zs_test_result = zs_agent.test(
        testing_dataset=test_data,
        prompt=zscot_predict_prompt_t14,
        schema=zs_testing_schema,
        label="n",
    )
    n_zs_test_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result/0716_n03_zs_test_800.csv",
        index=False,
    )  # t results are included as well
