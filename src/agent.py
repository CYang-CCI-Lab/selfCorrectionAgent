import os
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Union, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import time
import json
from fuzzywuzzy import fuzz
from openai import OpenAI
from prompt import system_instruction

# # Optional: Uncomment if using langsmith tracing
# if not load_dotenv(find_dotenv()):
#     raise Exception("Failed to load .env file")
# from langsmith import traceable
# from langsmith.wrappers import wrap_openai

class ChoiceAgent:
    """ the simplest agent, which is appropriate for zero-shot prompting
    """
    def __init__(self, client: OpenAI, model: str, 
                 prompt_template: str, choices: List, label: str) -> None:
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.choices = choices
        self.label = label

    def get_response_from_choices(self, messages: list, temperature:float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"guided_choice":self.choices},
            temperature = temperature
        )
        return response.choices[0].message.content

    def run(self, dataset: pd.DataFrame, temperature: float = 0.1) -> pd.DataFrame:
        pbar = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            report = row["text"]
            prompt = self.prompt_template.format(report=report)
            system_prompt = system_instruction+ "\n" + prompt
            messages = [{"role": "user", "content": system_prompt}]
            answer = self.get_response_from_choices(messages, temperature)
            dataset.loc[idx, f"zs_{self.label}_ans_str"] = answer
            pbar.update(1)
        pbar.close()

        return dataset

class ConditionalMemoryAgent:
    """ the implementation of memory agent, which learn memory from training set and
    utilize memory as contexts for testing set.
    """
    def __init__(self, client: OpenAI, model: str, 
                 prompt_template_dict: dict[str, str], schema_dict: dict, label: str) -> None:
        self.client = client
        self.model = model
        self.prompt_template_dict = prompt_template_dict
        self.validate_prompt_template()
        self.schema_dict = schema_dict
        self.validate_schema()
        self.memory = ""
        self.label = label

    def validate_prompt_template(self) -> None:
        keys = self.prompt_template_dict.keys()
        initial_prompt_exist = "initialized_prompt" in keys
        learning_prompt_exist = "learning_prompt" in keys
        testing_prompt_exist = "testing_prompt" in keys
        assert True == initial_prompt_exist == learning_prompt_exist == testing_prompt_exist, \
        "You should provide a dict with initialized_prompt, learning_prompt, and testing_prompt as keys."

    def validate_schema(self) -> None:
        keys = self.schema_dict.keys()
        learning_schema_exist = "learning_schema" in keys
        testing_schema_exist = "testing_schema" in keys
        assert True == learning_schema_exist == testing_schema_exist, \
        "You should provide a dict with learning_schema and testing_schema as keys."

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
        except json.JSONDecodeError:
            print("Error decoding JSON response")
            return None
        except openai.APITimeoutError:
            print("API Timeout Error")
            return None
        except Exception as e:
                print(f"An error occurred: {e}")
                return None
                
    def train(self, training_dataset: pd.DataFrame, temperature: float = 0.1, threshold: float = 80) -> pd.DataFrame:
        # only overide this function because the rest parts are the same
        pbar = tqdm(total=training_dataset.shape[0])
        parsing_error = 0
        num_update = 0
        for idx, row in training_dataset.iterrows():

            report = row["text"]

            if self.memory == "":
                prompt = self.prompt_template_dict["initialized_prompt"].format(report=report)
            else:
                prompt = self.prompt_template_dict["learning_prompt"].format(memory=self.memory, report=report)
            
            system_prompt = system_instruction+ "\n" + prompt
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, self.schema_dict["learning_schema"], temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                training_dataset.loc[idx, f"cmem_{self.label}_is_parsed"] = False
                continue

            training_dataset.loc[idx, f"cmem_{self.label}_is_parsed"] = True

            if self.memory == "":
                self.memory = json_output['rules']
            else:
                current_memory_str = "\n".join(self.memory)
                new_memory_str = "\n".join(json_output['rules'])
                training_dataset.loc[idx, f"cmem_{self.label}_edit_distance"] = fuzz.ratio(current_memory_str, new_memory_str)
                if fuzz.ratio(current_memory_str, new_memory_str) >= threshold :
                    self.memory = json_output['rules']
                    num_update += 1
                    training_dataset.loc[idx, f"cmem_{self.label}_is_updated"] = True
                else:
                    training_dataset.loc[idx, f"cmem_{self.label}_is_updated"] = False
            
            training_dataset.loc[idx, f"cmem_{self.label}_rules_str"] = "\n".join(json_output['rules'])
            training_dataset.loc[idx, f"cmem_{self.label}_memory_str"] =  "\n".join(self.memory)
            training_dataset.loc[idx, f"cmem_{self.label}_memory_len"] = len(self.memory)
            training_dataset.loc[idx, f"cmem_{self.label}_memory_str_len"] = len("\n".join(self.memory))
            
            training_dataset.loc[idx, f"cmem_{self.label}_ans_str"] = json_output['predictedStage']
            training_dataset.loc[idx, f"cmem_{self.label}_reasoning"] = json_output['reasoning']

            pbar.update(1)
        pbar.close()
        print(f"Number of memory updates: {num_update}")
        print(f"During training, number of parsing errors: {parsing_error}")

        return training_dataset

    def test(self, testing_dataset: pd.DataFrame, external_memory = "", temperature: float = 0.1) -> pd.DataFrame:
        pbar = tqdm(total=testing_dataset.shape[0])
        parsing_error = 0
    
        for idx, row in testing_dataset.iterrows():

            report = row["text"]
            if external_memory:
                self.memory = external_memory
            prompt = self.prompt_template_dict["testing_prompt"].format(memory=self.memory, report=report)
            system_prompt = system_instruction+ "\n" + prompt
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, self.schema_dict["testing_schema"], temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"cmem_{self.label}_is_parsed"] = False
                continue
            testing_dataset.loc[idx, f"cmem_{self.label}_is_parsed"] = True

            testing_dataset.loc[idx, f"cmem_{self.label}_reasoning"] = json_output['reasoning']
            testing_dataset.loc[idx, f"cmem_{self.label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)
        pbar.close()
        print(f"During testing, number of parsing errors: {parsing_error}")
        return testing_dataset
    
    #   @traceable(run_type="tool", name="dynamic_test")
    def dynamic_test(self, testing_dataset: pd.DataFrame, memory_tup: List[tuple], temperature: float = 0.1) -> pd.DataFrame:
        pbar = tqdm(total=testing_dataset.shape[0])
        parsing_error = 0
        for idx, row in testing_dataset.iterrows():

            report = row["text"]

            for num, memory in memory_tup:
                prompt = self.prompt_template_dict["testing_prompt"].format(memory=memory, report=report)
                system_prompt = system_instruction+ "\n" + prompt
                messages = [{"role": "user", "content": system_prompt}]

                json_output = self.get_schema_followed_response(messages, self.schema_dict["testing_schema"], temperature)

                if not json_output:
                    parsing_error += 1
                    print(f"Error at index: {idx}")
                    testing_dataset.loc[idx, f"cmem_{self.label}_{num}reports_is_parsed"] = False
                    continue
                
                testing_dataset.loc[idx, f"cmem_{self.label}_{num}reports_is_parsed"] = True
                testing_dataset.loc[idx, f"cmem_{self.label}_{num}reasoning"] = json_output['reasoning']
                testing_dataset.loc[idx, f"cmem_{self.label}_{num}reports_ans_str"] = json_output['predictedStage']
                
            pbar.update(1)
        pbar.close()
        print(f"During dynamic testing, number of parsing errors: {parsing_error}")
        return testing_dataset
    
    def zs_test(self, testing_dataset: pd.DataFrame, temperature: float = 0.1) -> pd.DataFrame:
        pbar = tqdm(total=testing_dataset.shape[0])
        parsing_error = 0
    
        for idx, row in testing_dataset.iterrows():

            report = row["text"]
        
            prompt = self.prompt_template_dict["testing_prompt"].format(report=report)
            system_prompt = system_instruction+ "\n" + prompt
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, self.schema_dict["testing_schema"], temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"zs_{self.label}_is_parsed"] = False
                continue

            testing_dataset.loc[idx, f"zs_{self.label}_is_parsed"] = True
            testing_dataset.loc[idx, f"zs_{self.label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)
        pbar.close()
        print(f"During zero-shot testing, number of parsing errors: {parsing_error}")
        return testing_dataset
    
    def zscot_test(self, testing_dataset: pd.DataFrame, temperature: float = 0.1) -> pd.DataFrame:
        pbar = tqdm(total=testing_dataset.shape[0])
        parsing_error = 0
    
        for idx, row in testing_dataset.iterrows():

            report = row["text"]
        
            prompt = self.prompt_template_dict["testing_prompt"].format(report=report)
            system_prompt = system_instruction+ "\n" + prompt
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, self.schema_dict["testing_schema"], temperature)

            if not json_output:
                parsing_error += 1
                print(f"Error at index: {idx}")
                testing_dataset.loc[idx, f"zscot_{self.label}_is_parsed"] = False
                continue
            testing_dataset.loc[idx, f"zscot_{self.label}_is_parsed"] = True
            testing_dataset.loc[idx, f"zscot_{self.label}_reasoning"] = json_output['reasoning']
            testing_dataset.loc[idx, f"zscot_{self.label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)
        pbar.close()
        print(f"During zero-shot cot testing, number of parsing errors: {parsing_error}")
        return testing_dataset
    
    def clear_memory(self):
        self.memory = ""
        print("Memory is cleared.")
    
