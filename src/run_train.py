from agent import *
from prompt import *
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

if not load_dotenv(find_dotenv()):
    raise Exception("Failed to load .env file")
from typing import Literal, List


class Response_T(BaseModel):
    reasoning: str = Field(
        description="Step-by-step explanation of how you interpreted the report to determine the T stage."
    )
    stage: Literal["T1", "T2", "T3", "T4"] = Field(
        description="The T stage determined from the report. Stage must be one of 'T1', 'T2', 'T3' or 'T4.'"
    )

class Response_RuleExtraction_T(BaseModel):
    error_analysis: str = Field(
        description="A detailed explanation of where the model's reasoning diverged from the correct reasoning that leads to the ground truth T stage. This should clearly articulate the mistake or oversight."
    )
    rules: List[str] = Field(
        description="A list of general, AJCC guideline-based rules inferred from the error analysis. These rules should help ensure correct T stage identification in similar scenarios."
    )

class Response_RuleRefinement_T(BaseModel):
    refined_rules: List[str] = Field(
        description="A consolidated, authoritative set of general AJCC-based rules for determining T stages in breast cancer. The list should be free of redundancies, contradictions, and case-specific details."
    )

schema_set = {"1": Response_T.model_json_schema(), "2": Response_RuleExtraction_T.model_json_schema(), "3": Response_RuleRefinement_T.model_json_schema()}


client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1", timeout=120.0)

if __name__ == "__main__":

    brca_report = pd.read_csv(
        "/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv"
    )
    brca_report = brca_report[brca_report["n"] != -1]

    training_set = set()
    gpt_df = pd.read_csv('/home/yl3427/cylab/selfCorrectionAgent/result/1112_t14_gpt_test_0_outof_8runs.csv').sort_values(by="patient_filename")
    llama_df = pd.read_csv('/home/yl3427/cylab/selfCorrectionAgent/result/1208_t14_llama3_kewltm_0_outof_10runs.csv').sort_values(by="patient_filename")
    for i in range(len(gpt_df)):
        label = gpt_df.loc[i, 't']
        gpt = gpt_df.loc[i, 'gpt4o_t_stage']
        llama = llama_df.loc[i, 't14_kewltm_t_pred']
        if (gpt != llama) and (gpt == f"T{label+1}"):
            training_set.add(gpt_df.loc[i, 'patient_filename'])
    
    training_data = brca_report[brca_report["patient_filename"].isin(training_set)][["patient_filename", "t", "text"]]
    test_data = brca_report[~brca_report["patient_filename"].isin(training_set)][["patient_filename", "t", "text"]]

    train_agent = Agent(
        client=client,
        model="meta-llama/Llama-3.3-70B-Instruct",
        label="t",
        schema=schema_set,
        test_name="t14_train"
    )

    result_training_data, rules, refined_rules = train_agent.train(training_data)
    result_training_data.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/1213_training_data.csv", index=False)
    with open("/home/yl3427/cylab/selfCorrectionAgent/result/1213_rules.txt", "w") as f:
        f.write("\n".join(rules))
    with open("/home/yl3427/cylab/selfCorrectionAgent/result/1213_refined_rules.txt", "w") as f:
        f.write("\n".join(refined_rules))

    test_agent = Agent(
        client=client,
        model="meta-llama/Llama-3.3-70B-Instruct",
        label="t",
        schema=schema_set['1'],
        test_name="t14_test"
    )

    result_test_data = test_agent.test(test_data, prompt_test_with_rules_t14, "\n".join(refined_rules))
    result_test_data.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/1213_test_data_with_refined_ltm.csv", index=False)

    result_test_data_base = test_agent.test(test_data, prompt_zscot_t14)
    result_test_data_base.to_csv("/home/yl3427/cylab/selfCorrectionAgent/result/1213_test_data_base.csv", index=False)

