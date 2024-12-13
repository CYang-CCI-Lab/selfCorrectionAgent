import os
import time
import json
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from prompt import *
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
from pathlib import Path

env_path = Path.home()
load_dotenv(dotenv_path=env_path / ".env")


class Agent:
    def __init__(
        self, client: OpenAI, model: str, label: str, schema: dict, test_name: str
    ) -> None:
        self.client = client
        self.model = model
        self.label = label
        self.schema = schema
        self.test_name = test_name

    def get_schema_followed_response(self, messages: list) -> Union[Dict, None]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"guided_json": self.schema},
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def train(
        self,
        training_dataset: pd.DataFrame,
    ):
        parsing_error = 0
        pbar = tqdm(total=training_dataset.shape[0])

        for idx, row in training_dataset.iterrows():
            report = row["text"]
            if self.label == "t":
                filled_prompt = rag_t14.format(report=report)
            else:
                filled_prompt = rag_n03.format(report=report)

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": filled_prompt},
            ]

            json_output = self.get_schema_followed_response(messages)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                training_dataset.loc[idx, f"{self.test_name}_{self.label}_is_parsed"] = (
                    False
                )
                continue

            training_dataset.loc[idx, f"{self.test_name}_{self.label}_is_parsed"] = True
            training_dataset.loc[idx, f"{self.test_name}_{self.label}_reasoning"] = (
                json_output["reasoning"]
            )
            training_dataset.loc[idx, f"{self.test_name}_{self.label}_pred"] = (
                json_output["stage"]
            )
            if f"T{training_dataset.loc[idx, self.label]+1}" != json_output["stage"]:
                feedback_prompt
                training_dataset.loc[idx, f"{self.test_name}_{self.label}_correct"] = True
            response = response.choices[0].message.content
            messages.append({"role": "assistant", "content": response})
            training_dataset.loc[idx, f"generated_assess"] = response

            if ltm:
                filled_prompt = ltm_cot2_test.format(
                    disease=self.disease,
                    list_of_rules=ltm,
                    subjective_section=subj,
                    objective_section=obj,
                )
                column_prefix = f"{self.disease}_with_ltm"
            else:
                filled_prompt = without_ltm_cot2_test.format(
                    disease=self.disease, subjective_section=subj, objective_section=obj
                )
                column_prefix = f"{self.disease}_without_ltm"

            messages.append({"role": "user", "content": filled_prompt})

            response = self.get_response(messages, schema)

            if not response:
                parsing_error += 1
                print(f"Error at index: {idx}")
                training_dataset.loc[idx, f"{column_prefix}_is_parsed"] = False
                continue

            training_dataset.loc[idx, f"{column_prefix}_is_parsed"] = True
            training_dataset.loc[idx, f"{column_prefix}_answer"] = response["answer"]

            pbar.update(1)
        pbar.close()
        print(f"Total parsing errors: {parsing_error}")
        return training_dataset


    def test(
        self,
        testing_dataset: pd.DataFrame,
        prompt: str,
        context: Optional[str] = None,
    ) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            if context:
                filled_prompt = prompt.format(report=report, context=context)
            else:
                filled_prompt = prompt.format(report=report)

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": filled_prompt},
            ]

            json_output = self.get_schema_followed_response(messages)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"{self.test_name}_{self.label}_is_parsed"] = (
                    False
                )
                continue

            testing_dataset.loc[idx, f"{self.test_name}_{self.label}_is_parsed"] = True
            testing_dataset.loc[idx, f"{self.test_name}_{self.label}_reasoning"] = (
                json_output["reasoning"]
            )
            testing_dataset.loc[idx, f"{self.test_name}_{self.label}_pred"] = (
                json_output["stage"]
            )

            pbar.update(1)

        pbar.close()
        print(f"Total number of parsing errors: {parsing_error}")
        return testing_dataset
