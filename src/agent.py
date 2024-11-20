import os
import time
import json
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from openai import OpenAI
from prompt import *
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
from pathlib import Path

env_path = Path.home()
load_dotenv(dotenv_path=env_path / ".env")


class ZSAgent:
    """
    ZSAgent evaluates testing set using zero-shot.
    ZSAgent is a base class for other agents.
    """

    def __init__(self, client: OpenAI, model: str, label: str) -> None:
        self.client = client
        self.model = model
        self.label = label

    def get_schema_followed_response(
        self, messages: list, schema: dict, temperature: float
    ) -> Union[Dict, None]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={
                    "guided_json": schema,
                    # "guided_regex": "T[1-4]|t[1-4]|N[0-3]|n[0-3]",
                    # "guided_choice": ["T1", "T2", "T3", "T4"] # ["N0", "N1", "N2", "N3"]
                },
                temperature=temperature,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def test(
        self,
        testing_dataset: pd.DataFrame,
        prompt: str,
        schema: dict,
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            filled_prompt = prompt.format(report=report)
            messages = [{"role": "user", "content": filled_prompt}]

            json_output = self.get_schema_followed_response(
                messages, schema, temperature
            )

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"zs_{self.label}_is_parsed"] = False
                continue

            testing_dataset.loc[idx, f"zs_{self.label}_is_parsed"] = True
            testing_dataset.loc[idx, f"zs_{self.label}_reasoning"] = json_output[
                "reasoning"
            ]
            testing_dataset.loc[idx, f"zs_{self.label}_ans_str"] = json_output["stage"]

            pbar.update(1)

        pbar.close()
        print(f"During zero-shot testing, number of parsing errors: {parsing_error}")
        return testing_dataset


class ZSCOTAgent(ZSAgent):
    """
    ZSCOTAgent evaluates testing set using zero-shot cot.
    """

    def __init__(self, client: OpenAI, model: str, label: str) -> None:
        super().__init__(client, model, label)

    def test(
        self,
        testing_dataset: pd.DataFrame,
        prompt: str,
        schema: dict,
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            filled_prompt = prompt.format(report=report)
            messages = [{"role": "user", "content": filled_prompt}]

            json_output = self.get_schema_followed_response(
                messages, schema, temperature
            )

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"zscot_{self.label}_is_parsed"] = False
                continue

            testing_dataset.loc[idx, f"zscot_{self.label}_is_parsed"] = True
            testing_dataset.loc[idx, f"zscot_{self.label}_reasoning"] = json_output[
                "reasoning"
            ]
            testing_dataset.loc[idx, f"zscot_{self.label}_ans_str"] = json_output[
                "stage"
            ]

            pbar.update(1)

        pbar.close()
        print(
            f"During zero-shot cot testing, number of parsing errors: {parsing_error}"
        )
        return testing_dataset


class KEPATrainAgent(ZSAgent):
    """
    KEPATrainAgent learns memory from training set.
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)
        self.memory = ""

    def validate_prompt_template(self, prompt_template_dict) -> None:
        keys = prompt_template_dict.keys()
        initial_prompt_exist = "initial_prompt" in keys
        subsequent_prompt_exist = "subsequent_prompt" in keys
        assert (
            initial_prompt_exist == subsequent_prompt_exist
        ), "You should provide a dict with 'initial_prompt' and 'subsequent_prompt' as keys."

    def train(
        self,
        training_dataset: pd.DataFrame,
        prompt_template_dict: dict,
        schema: dict,
        label: str,
        threshold: float = 80,
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        self.validate_prompt_template(prompt_template_dict)
        parsing_error = 0
        pbar = tqdm(total=training_dataset.shape[0])
        num_update = 0

        for idx, row in training_dataset.iterrows():
            report = row["text"]

            if self.memory == "":
                prompt = (
                    system_instruction
                    + "\n"
                    + prompt_template_dict["initial_prompt"].format(report=report)
                )
            else:
                prompt = (
                    system_instruction
                    + "\n"
                    + prompt_template_dict["subsequent_prompt"].format(
                        memory=self.memory, report=report
                    )
                )

            messages = [{"role": "user", "content": prompt}]

            json_output = self.get_schema_followed_response(
                messages, schema, temperature
            )

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                training_dataset.loc[idx, f"kepa_{label}_is_parsed"] = False
                continue

            training_dataset.loc[idx, f"kepa_{label}_is_parsed"] = True

            if self.memory == "":
                self.memory = json_output["rules"]
            else:
                current_memory_str = "\n".join(self.memory)
                new_memory_str = "\n".join(json_output["rules"])
                training_dataset.loc[idx, f"kepa_{label}_edit_distance"] = fuzz.ratio(
                    current_memory_str, new_memory_str
                )
                if fuzz.ratio(current_memory_str, new_memory_str) >= threshold:
                    self.memory = json_output["rules"]
                    num_update += 1
                    training_dataset.loc[idx, f"kepa_{label}_is_updated"] = True
                else:
                    training_dataset.loc[idx, f"kepa_{label}_is_updated"] = False

            training_dataset.loc[idx, f"kepa_{label}_rules_str"] = "\n".join(
                json_output["rules"]
            )
            training_dataset.loc[idx, f"kepa_{label}_memory_str"] = "\n".join(
                self.memory
            )
            training_dataset.loc[idx, f"kepa_{label}_memory_len"] = len(self.memory)
            training_dataset.loc[idx, f"kepa_{label}_memory_str_len"] = len(
                "\n".join(self.memory)
            )
            training_dataset.loc[idx, f"kepa_{label}_ans_str"] = json_output["stage"]
            training_dataset.loc[idx, f"kepa_{label}_reasoning"] = json_output[
                "reasoning"
            ]

            pbar.update(1)

        pbar.close()
        print(f"Number of memory updates: {num_update}")
        print(f"During training, number of parsing errors: {parsing_error}")
        return training_dataset


class KEPATestAgent(ZSAgent):
    """
    KEPATestAgent evaluates testing set with memory learned from training set.
    """

    def __init__(self, client: OpenAI, model: str, label: str) -> None:
        super().__init__(client, model, label)

    def test(
        self,
        testing_dataset: pd.DataFrame,
        prompt: str,
        schema: dict,
        memory: str,
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            filled_prompt = prompt.format(memory=memory, report=report)
            messages = [{"role": "user", "content": filled_prompt}]

            json_output = self.get_schema_followed_response(
                messages, schema, temperature
            )

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"kepa_{self.label}_is_parsed"] = False
                continue

            testing_dataset.loc[idx, f"kepa_{self.label}_is_parsed"] = True
            testing_dataset.loc[idx, f"kepa_{self.label}_reasoning"] = json_output[
                "reasoning"
            ]
            testing_dataset.loc[idx, f"kepa_{self.label}_ans_str"] = json_output[
                "stage"
            ]

            pbar.update(1)

        pbar.close()
        print(f"During testing, number of parsing errors: {parsing_error}")
        return testing_dataset

    def dynamic_test(
        self,
        testing_dataset: pd.DataFrame,
        prompt: str,
        schema: dict,
        memory_tup: List[tuple],
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]

            for num, memory in memory_tup:
                filled_prompt = prompt.format(memory=memory, report=report)
                messages = [{"role": "user", "content": filled_prompt}]

                json_output = self.get_schema_followed_response(
                    messages, schema, temperature
                )

                if not json_output:
                    parsing_error += 1
                    print(f"Error at index: {idx}")
                    testing_dataset.loc[
                        idx, f"kepa_{self.label}_{num}reports_is_parsed"
                    ] = False
                    continue

                testing_dataset.loc[
                    idx, f"kepa_{self.label}_{num}reports_is_parsed"
                ] = True
                testing_dataset.loc[idx, f"kepa_{self.label}_{num}reasoning"] = (
                    json_output["reasoning"]
                )
                testing_dataset.loc[idx, f"kepa_{self.label}_{num}reports_ans_str"] = (
                    json_output["stage"]
                )

            pbar.update(1)

        pbar.close()
        print(f"During dynamic testing, number of parsing errors: {parsing_error}")
        return testing_dataset


class PostHocVerificationAgent(ZSAgent):
    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)

    def verify(
        self,
        testing_dataset: pd.DataFrame,
        reasoning_column: str,
        prompt: str,
        schema: dict,
        label: str,
        temperature: float = 0.1,
    ) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            reasoning = row[reasoning_column]
            filled_prompt = prompt.format(reasoning=reasoning)
            messages = [{"role": "user", "content": filled_prompt}]

            json_output = self.get_schema_followed_response(
                messages, schema, temperature
            )

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                continue

            testing_dataset.loc[idx, f"{label}_evaluation"] = json_output["evaluation"]

            pbar.update(1)

        pbar.close()
        print(f"During verifying, number of parsing errors: {parsing_error}")
        return testing_dataset


class StageEnum_T(str, Enum):
    """TNM classification - T staging."""

    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"


class StageEnum_N(str, Enum):
    """TNM classification - N staging."""

    N0 = "N0"
    N1 = "N1"
    N2 = "N2"
    N3 = "N3"


class Response_T(BaseModel):
    reasoning: str = Field(
        description="Step-by-step explanation of how you interpreted the report to determine the T stage."
    )
    stage: StageEnum_T = Field(description="The T stage determined from the report.")


class Response_N(BaseModel):
    reasoning: str = Field(
        description="Step-by-step explanation of how you interpreted the report to determine the N stage."
    )
    stage: StageEnum_N = Field(description="The N stage determined from the report.")


class GPTAgent:
    def __init__(self, label: str) -> None:
        self.client = OpenAI()
        self.label = label

    def get_response(self, prompt):
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=0,
            response_format=Response_T if self.label.lower() == "t" else Response_N,
        )

        return response.choices[0].message.parsed

    def test(self, testing_dataset: pd.DataFrame, memory: str) -> pd.DataFrame:

        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            structured_prompt = (
                testing_predict_prompt_t14.format(memory=memory, report=report)
                if self.label.lower() == "t"
                else testing_predict_prompt_n03.format(memory=memory, report=report)
            )
            response = self.get_response(structured_prompt)

            testing_dataset.loc[idx, f"gpt4o_{self.label}_reasoning"] = (
                response.reasoning
            )
            testing_dataset.loc[idx, f"gpt4o_{self.label}_stage"] = response.stage

            pbar.update(1)
        pbar.close()
        print(f"During testing, number of parsing errors: {parsing_error}")
        return testing_dataset
