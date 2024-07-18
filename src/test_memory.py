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

    run = 8

    # test data (common for both t14 and n03)
    # test_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_test_{run}.csv")[['Unnamed: 0', 'patient_filename', 't', 'text', 'n']]

    # zs data (the entire 800 reports)
    brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
    brca_report = brca_report[brca_report["n"]!=-1]
    test_data = brca_report[['patient_filename', 't', 'text', 'n']]

    # zs
    memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
                                                "learning_prompt":subsequent_predict_prompt_t14,
                                                "testing_prompt":zs_predict_prompt_t14},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "t")

    test_result = memory_agent_t14.zs_test(testing_dataset=test_data)
    test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/0716/t14_zs_test_800.csv", index=False)

    memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
                                                "learning_prompt":subsequent_predict_prompt_n03,
                                                "testing_prompt":zs_predict_prompt_n03},
                        schema_dict={"learning_schema":training_schema,
                                        "testing_schema":testing_schema},
                                        label = "n")
    
    test_result = memory_agent_n03.zs_test(testing_dataset=test_data)
    test_result.to_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/0716/n03_zs_test_800.csv", index=False)


    # # t14 training data to extract memory
    # train_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_{run}.csv")

    # memory_dict_t = {}
    # for idx, row in train_data.iterrows():
    #     # if row["cmem_t_is_updated"] == True:
    #     memory_dict_t[f"{idx+1}"] = row['cmem_t_memory_str']
    # print(memory_dict_t['30'])

    # memory_agent_t14 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #                     prompt_template_dict={"initialized_prompt":initial_predict_prompt_t14,
    #                                             "learning_prompt":subsequent_predict_prompt_t14,
    #                                             "testing_prompt":testing_predict_prompt_t14},
    #                     schema_dict={"learning_schema":training_schema,
    #                                     "testing_schema":testing_schema},
    #                                     label = "t")

    # test_result = memory_agent_t14.test(testing_dataset=test_data, num=run, external_memory=memory_dict_t['30'])
    # test_result.to_csv(f"t14_test_{run}_memory_at_30", index=False)


    # # n03 training data to extract memory
    # train_data = pd.read_csv(f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_{run}.csv")

    # memory_dict_n = {}
    # for idx, row in train_data.iterrows():
    #     # if row["cmem_t_is_updated"] == True:
    #     memory_dict_n[f"{idx+1}"] = row['cmem_n_memory_str']
    # print(memory_dict_n['50'])

    
    # memory_agent_n03 = ConditionalMemoryAgent(client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #                     prompt_template_dict={"initialized_prompt":initial_predict_prompt_n03,
    #                                             "learning_prompt":subsequent_predict_prompt_n03,
    #                                             "testing_prompt":testing_predict_prompt_n03},
    #                     schema_dict={"learning_schema":training_schema,
    #                                     "testing_schema":testing_schema},
    #                                     label = "n")
    
    # test_result = memory_agent_n03.test(testing_dataset=test_data, num=run, external_memory=memory_dict_n['50'])
    # test_result.to_csv(f"n03_test_{run}_memory_at_50", index=False)

