from agent import KEPATrainAgent
from prompt import *
from openai import OpenAI
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Union
from datetime import datetime


class TrainingResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage")
    rules: List[str] = Field(description="list of rules")


if __name__ == "__main__":
    client = OpenAI(api_key="empty", base_url="http://localhost:8000/v1")

    training_schema = TrainingResponse.model_json_schema()

    for run in range(10):

        # t14
        t_train_file_path = (
            f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_train_{run}.csv"
        )
        t_train_data = pd.read_csv(t_train_file_path)[["patient_filename", "text", "t"]]

        memory_agent_t14 = KEPATrainAgent(
            client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )

        t_train_result = memory_agent_t14.train(
            t_train_data,
            prompt_template_dict={
                "initial_prompt": initial_predict_prompt_t14,
                "subsequent_prompt": subsequent_predict_prompt_t14,
            },
            schema=training_schema,
            label="t",
        )
        t_train_result.to_csv(
            f"/home/yl3427/cylab/selfCorrectionAgent/result/t14_memory_dataset{run}.csv",
            index=False,
        )

        # n03
        n_train_file_path = (
            f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_train_{run}.csv"
        )
        n_train_data = pd.read_csv(n_train_file_path)[["patient_filename", "text", "n"]]

        memory_agent_n03 = KEPATrainAgent(
            client=client, model="mistralai/Mixtral-8x7B-Instruct-v0.1"
        )

        n_train_result = memory_agent_n03.train(
            n_train_data,
            prompt_template_dict={
                "initial_prompt": initial_predict_prompt_n03,
                "subsequent_prompt": subsequent_predict_prompt_n03,
            },
            schema=training_schema,
            label="n",
        )
        n_train_result.to_csv(
            f"/home/yl3427/cylab/selfCorrectionAgent/result/n03_memory_dataset{run}.csv",
            index=False,
        )
