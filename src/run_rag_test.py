from agent import *
from prompt import *
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
if not load_dotenv(find_dotenv()):
    raise Exception("Failed to load .env file")
from typing import Literal

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

with open("/home/yl3427/cylab/rag_tnm/src/context.json", "r") as f:
    context = json.load(f)

rag_raw_t14 = context["rag_raw_t14"]
rag_raw_n03 = context["rag_raw_n03"]
ltm_zs_t14 = context["ltm_zs_t14"]
ltm_zs_n03 = context["ltm_zs_n03"]
ltm_rag1_t14 = context["ltm_rag1_t14"]
ltm_rag1_n03 = context["ltm_rag1_n03"]
ltm_rag2_t14 = context["ltm_rag2_t14"]
ltm_rag2_n03 = context["ltm_rag2_n03"]


if __name__ == "__main__":
    client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)

    brca_report = pd.read_csv(
        "/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv"
    )
    brca_report = brca_report[brca_report["n"] != -1]
    test_data = brca_report[["patient_filename", "t", "text", "n"]]

    #### T14
    testing_schema = Response_T.model_json_schema()

    zscot_agent_t14 = ZSCOTAgent(
        client=client, model="m42-health/Llama3-Med42-70B", label="t"
    )

    t_zscot_test_result = zscot_agent_t14.test(
        testing_dataset=test_data,
        prompt=zscot_predict_prompt_t14,
        schema=testing_schema,
    )
    t_zscot_test_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result/1118_t14_med42_v2_test_800.csv",
        index=False,
    )

    #### N03
    testing_schema = Response_N.model_json_schema()

    zscot_agent_n03 = ZSCOTAgent(
        client=client, model="m42-health/Llama3-Med42-70B", label="n"
    )
    n_zscot_test_result = zscot_agent_n03.test(
        testing_dataset=test_data,
        prompt=zscot_predict_prompt_n03,
        schema=testing_schema,
    )
    n_zscot_test_result.to_csv(
        f"/home/yl3427/cylab/selfCorrectionAgent/result/1118_n03_med42_v2_test_800.csv",
        index=False,
    )
