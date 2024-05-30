from prompt import system_instruction
import openai
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time
import json
from typing import List, Dict, Union
from fuzzywuzzy import fuzz

class ChoiceAgent:
    """ the simplest agent, which is appropriate for zero-shot prompting
    """
    def __init__(self, client: OpenAI, model: str, 
                 prompt_template: str, choices: dict, label: str) -> None:
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

    def run(self, dataset: pd.DataFrame, temperature: float = 0.0) -> pd.DataFrame:
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

class MemoryAgent:
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

    def get_schema_followed_response(self, messages: list, schema:dict, temperature:float) -> Union[Dict, None]:
        num_attempt = 2
        for attempt in range(num_attempt):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body={"guided_json":schema},
                    temperature = temperature
                )
                return json.loads(response.choices[0].message.content.replace("\\", "\\\\"))
            except json.JSONDecodeError:
                print("Error decoding JSON response")
                return None
            except openai.APITimeoutError:
                if attempt < (num_attempt -1):
                    wait_time = 2 * (attempt + 1)
                    print(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Request faild.")
                    return None
        
    def train(self, training_dataset: pd.DataFrame, temperature: float = 0.0) -> pd.DataFrame:
        pbar = tqdm(total=training_dataset.shape[0])
        parsing_error = 0
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
                continue
            
            self.memory = json_output['rules']
            training_dataset.loc[idx, f"mem_{self.label}_reasoning"] = json_output['reasoning']
            training_dataset.loc[idx, f"mem_{self.label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)
        pbar.close()
        print(f"During training, number of parsing errors: {parsing_error}")
        return training_dataset
    
    def test(self, testing_dataset: pd.DataFrame, temperature: float = 0.0) -> pd.DataFrame:
        pbar = tqdm(total=testing_dataset.shape[0])
        parsing_error = 0
        for idx, row in testing_dataset.iterrows():

            report = row["text"]

            prompt = self.prompt_template_dict["testing_prompt"].format(memory=self.memory, report=report)
            system_prompt = system_instruction+ "\n" + prompt
            messages = [{"role": "user", "content": system_prompt}]

            json_output = self.get_schema_followed_response(messages, self.schema_dict["testing_schema"], temperature)

            if not json_output:
                parsing_error += 1
                continue
            
            testing_dataset.loc[idx, f"mem_{self.label}_reasoning"] = json_output['reasoning']
            testing_dataset.loc[idx, f"mem_{self.label}_ans_str"] = json_output['predictedStage']
            
            pbar.update(1)
        pbar.close()
        print(f"During testing, number of parsing errors: {parsing_error}")
        return testing_dataset
    

class ConditionalMemoryAgent(MemoryAgent):
  
  def __init__(self, client: OpenAI, model: str, 
                 prompt_template_dict: dict[str, str], schema_dict: dict, label: str) -> None:
    # inherit all properties and methods from MemoryAgent
    super().__init__(client, model, prompt_template_dict, schema_dict, label)
    
  def train(self, training_dataset: pd.DataFrame, temperature: float = 0.0, threshold: float = 80) -> pd.DataFrame:
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
            continue
        
        if self.memory == "":
           self.memory = json_output['rules']
        else:
          current_memory_str = "\n".join(self.memory)
          new_memory_str = "\n".join(json_output['rules'])
          if fuzz.ratio(current_memory_str, new_memory_str) >= threshold :
            self.memory = json_output['rules']
            num_update += 1

        training_dataset.loc[idx, f"cmem_{self.label}_reasoning"] = json_output['reasoning']
        training_dataset.loc[idx, f"cmem_{self.label}_ans_str"] = json_output['predictedStage']
        training_dataset.loc[idx, f"cmem_{self.label}_num_update"] = num_update
        
        pbar.update(1)
    pbar.close()
    print(f"Number of memory updates: {num_update}")
    print(f"During training, number of parsing errors: {parsing_error}")
    return training_dataset

  def test(self, testing_dataset: pd.DataFrame, temperature: float = 0.0) -> pd.DataFrame:
    pbar = tqdm(total=testing_dataset.shape[0])
    parsing_error = 0
    for idx, row in testing_dataset.iterrows():

        report = row["text"]

        prompt = self.prompt_template_dict["testing_prompt"].format(memory=self.memory, report=report)
        system_prompt = system_instruction+ "\n" + prompt
        messages = [{"role": "user", "content": system_prompt}]

        json_output = self.get_schema_followed_response(messages, self.schema_dict["testing_schema"], temperature)

        if not json_output:
            parsing_error += 1
            continue
        
        # testing_dataset.loc[idx, f"cmem_{self.label}_reasoning"] = json_output['reasoning']
        testing_dataset.loc[idx, f"cmem_{self.label}_ans_str"] = json_output['predictedStage']
        
        pbar.update(1)
    pbar.close()
    print(f"During testing, number of parsing errors: {parsing_error}")
    return testing_dataset