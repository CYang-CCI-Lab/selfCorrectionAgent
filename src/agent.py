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

            messages = [{"role": "user", "content": filled_prompt}]

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
