import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from openai import OpenAI
from prompt import system_instruction

# # Optional: Uncomment if using langsmith tracing
# if not load_dotenv(find_dotenv()):
#     raise Exception("Failed to load .env file")
# from langsmith import traceable
# from langsmith.wrappers import wrap_openai

class ZSAgent:
    """
    ZSAgent evaluates testing set using zero-shot.
    ZSAgent is a base class for other agents.
    """
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    # @traceable(run_type="llm", name="get_schema_followed_response")
    def get_schema_followed_response(self, messages: list, schema:dict, temperature:float) -> Union[Dict, None]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"guided_json":schema},
                temperature = temperature
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
                
    def test(self, testing_dataset: pd.DataFrame, prompt: str, schema: dict, label: str, temperature: float = 0.1) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            system_prompt = system_instruction+ "\n" + prompt.format(report=report)
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, schema, temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"zs_{label}_is_parsed"] = False
                continue

            testing_dataset.loc[idx, f"zs_{label}_is_parsed"] = True
            testing_dataset.loc[idx, f"zs_{label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)

        pbar.close()
        print(f"During zero-shot testing, number of parsing errors: {parsing_error}")
        return testing_dataset
    
class ZSCOTAgent(ZSAgent):
    """
    ZSCOTAgent evaluates testing set using zero-shot cot.
    """
    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)
        
    def test(self, testing_dataset: pd.DataFrame, prompt: str, schema: dict, label: str, temperature: float = 0.1) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            system_prompt = system_instruction+ "\n" + prompt.format(report=report)
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, schema, temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"zscot_{label}_is_parsed"] = False
                continue

            testing_dataset.loc[idx, f"zscot_{label}_is_parsed"] = True
            testing_dataset.loc[idx, f"zscot_{label}_reasoning"] = json_output['reasoning']
            testing_dataset.loc[idx, f"zscot_{label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)

        pbar.close()
        print(f"During zero-shot cot testing, number of parsing errors: {parsing_error}")
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
        assert initial_prompt_exist == subsequent_prompt_exist, \
        "You should provide a dict with 'initial_prompt' and 'subsequent_prompt' as keys."
                
    def train(self, training_dataset: pd.DataFrame, prompt_template_dict: dict, schema: dict, label: str, threshold: float = 80, temperature: float = 0.1) -> pd.DataFrame:
        self.validate_prompt_template(prompt_template_dict)
        parsing_error = 0
        pbar = tqdm(total=training_dataset.shape[0])
        num_update = 0

        for idx, row in training_dataset.iterrows():
            report = row["text"]

            if self.memory == "":
                prompt = system_instruction+ "\n" + prompt_template_dict["initial_prompt"].format(report=report)
            else:
                prompt = system_instruction+ "\n" + prompt_template_dict["subsequent_prompt"].format(memory=self.memory, report=report)
            
            messages = [{"role": "user", "content": prompt}]

            json_output = self.get_schema_followed_response(messages, schema, temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                training_dataset.loc[idx, f"cmem_{label}_is_parsed"] = False
                continue

            training_dataset.loc[idx, f"cmem_{label}_is_parsed"] = True

            if self.memory == "":
                self.memory = json_output['rules']
            else:
                current_memory_str = "\n".join(self.memory)
                new_memory_str = "\n".join(json_output['rules'])
                training_dataset.loc[idx, f"cmem_{label}_edit_distance"] = fuzz.ratio(current_memory_str, new_memory_str)
                if fuzz.ratio(current_memory_str, new_memory_str) >= threshold :
                    self.memory = json_output['rules']
                    num_update += 1
                    training_dataset.loc[idx, f"cmem_{label}_is_updated"] = True
                else:
                    training_dataset.loc[idx, f"cmem_{label}_is_updated"] = False
            
            training_dataset.loc[idx, f"cmem_{label}_rules_str"] = "\n".join(json_output['rules'])
            training_dataset.loc[idx, f"cmem_{label}_memory_str"] =  "\n".join(self.memory)
            training_dataset.loc[idx, f"cmem_{label}_memory_len"] = len(self.memory)
            training_dataset.loc[idx, f"cmem_{label}_memory_str_len"] = len("\n".join(self.memory))
            training_dataset.loc[idx, f"cmem_{label}_ans_str"] = json_output['predictedStage']
            training_dataset.loc[idx, f"cmem_{label}_reasoning"] = json_output['reasoning']

            pbar.update(1)

        pbar.close()
        print(f"Number of memory updates: {num_update}")
        print(f"During training, number of parsing errors: {parsing_error}")
        return training_dataset
    

class KEPATestAgent(ZSAgent):
    """
    KEPATestAgent evaluates testing set with memory learned from training set.
    """
    def __init__(self, client: OpenAI, model: str) -> None:
        super().__init__(client, model)

    def test(self, testing_dataset: pd.DataFrame, prompt: str, schema: dict, label: str, memory: str, temperature: float = 0.1) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]
            system_prompt = system_instruction+ "\n" + prompt.format(memory=memory, report=report)
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, schema, temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"cmem_{label}_is_parsed"] = False
                continue

            testing_dataset.loc[idx, f"cmem_{label}_is_parsed"] = True
            testing_dataset.loc[idx, f"cmem_{label}_reasoning"] = json_output['reasoning']
            testing_dataset.loc[idx, f"cmem_{label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)

        pbar.close()
        print(f"During testing, number of parsing errors: {parsing_error}")
        return testing_dataset
    
    #   @traceable(run_type="tool", name="dynamic_test")
    def dynamic_test(self, testing_dataset: pd.DataFrame, prompt: str, schema: dict, label: str, memory_tup: List[tuple], temperature: float = 0.1) -> pd.DataFrame:
        parsing_error = 0
        pbar = tqdm(total=testing_dataset.shape[0])

        for idx, row in testing_dataset.iterrows():
            report = row["text"]

            for num, memory in memory_tup:
                system_prompt = system_instruction+ "\n" + prompt.format(memory=memory, report=report)
                messages = [{"role": "user", "content": system_prompt}]

                json_output = self.get_schema_followed_response(messages, schema, temperature)

                if not json_output:
                    parsing_error += 1
                    print(f"Error at index: {idx}")
                    testing_dataset.loc[idx, f"cmem_{label}_{num}reports_is_parsed"] = False
                    continue
                
                testing_dataset.loc[idx, f"cmem_{label}_{num}reports_is_parsed"] = True
                testing_dataset.loc[idx, f"cmem_{label}_{num}reasoning"] = json_output['reasoning']
                testing_dataset.loc[idx, f"cmem_{label}_{num}reports_ans_str"] = json_output['predictedStage']
                
            pbar.update(1)

        pbar.close()
        print(f"During dynamic testing, number of parsing errors: {parsing_error}")
        return testing_dataset