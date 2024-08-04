from agent import *
from prompt import *
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel, Field


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

    brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
    brca_report = brca_report[brca_report["n"]!=-1]
    test_data = brca_report[['patient_filename', 't', 'text', 'n']]

    zscot_agent = ZSCOTAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1")
      
    # zscot test for t14
    t_zscot_test_result = zscot_agent.test(testing_dataset=test_data, prompt=zscot_predict_prompt_t14, schema=testing_schema, label= 't')
    t_zscot_test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/0716_t14_zscot_test_800.csv", index=False)

    # zscot test for n03
    n_zscot_test_result = zscot_agent.test(testing_dataset=test_data, prompt=zscot_predict_prompt_n03, schema=testing_schema, label= 'n')
    n_zscot_test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/0716_n03_zscot_test_800.csv", index=False) # t results are included as well

        
    