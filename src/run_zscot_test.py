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

    brca_report = pd.read_csv(
        "/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv"
    )
    brca_report = brca_report[brca_report["n"] != -1]
    test_data = brca_report[["patient_filename", "t", "text", "n"]]

    #### T14
    t_schema = Response_T.model_json_schema()

    t_zscot_agent = Agent(
        client=client, 
        model="meta-llama/Llama-3.3-70B-Instruct", 
        label="t",
        schema=t_schema,
        test_name="t14_zscot"
    )

    t_zscot_result = t_zscot_agent.test(
        testing_dataset=test_data,
        prompt=zscot_t14
    )
    t_zscot_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result/1210_t14_llama3_zscot.csv",
        index=False,
    )
    t_zscot_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result_backup/1210_t14_llama3_zscot.csv",
        index=False,
    )

    #### N03
    n_schema = Response_N.model_json_schema()

    n_zscot_agent = Agent(
        client=client, 
        model="meta-llama/Llama-3.3-70B-Instruct", 
        label="n",
        schema=n_schema,
        test_name="n03_zscot"
    )

    n_zscot_result = n_zscot_agent.test(
        testing_dataset=test_data,
        prompt=zscot_n03
    )
    n_zscot_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result/1210_n03_llama3_zscot.csv",
        index=False,
    )
    n_zscot_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result_backup/1210_n03_llama3_zscot.csv",
        index=False,
    )