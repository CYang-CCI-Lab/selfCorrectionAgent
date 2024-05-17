import pandas as pd
import openai
from openai import OpenAI
from tqdm import tqdm
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
from typing import List
import json
import time
import os
from dotenv import load_dotenv
from prompt import *

def is_updated(old_memory, new_memory, threshold):
    old_str = "\n".join(old_memory)
    new_str = "\n".join(new_memory)
    if fuzz.ratio(old_str, new_str) >= threshold : 
        return True # update memory
    else:
        return False

def plot_in_box(lines):
    max_length = max(len(line) for line in lines if isinstance(line, str))
    print('-' * (max_length + 4))
    for line in lines:
        if "\n" in line:
            parts = line.split("\n")
            for part in parts:
                print(f"| {part.ljust(max_length)} |")
        else:
            print(f"| {line.ljust(max_length)} |")
    print('-' * (max_length + 4))

def get_response(client, index, prompt, schema, num_attempt = 5):
    for attempt in range(num_attempt):
        try:
            response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{"role": "user", "content": prompt}],
                extra_body={"guided_json":schema},
                temperature = 0.0
            )
            return json.loads(response.choices[0].message.content.replace("\\", "\\\\"))
        except openai.APITimeoutError as e:
            if attempt < (num_attempt - 1):
                wait_time = 2 * (attempt + 1)
                print(f"At {index}, Request timed out. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Request failed.")
                raise e
        except json.JSONDecodeError as e:
            print(f"At {index}, Error decoding JSON response: {response.choices[0].message.content}")
            return None

class BaseResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")

class Response(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage") # 3
    reasoning: str = Field(description="reasoning to support predicted cancer stage") # 2
    rules: List[str] = Field(description="list of rules") # 1



if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
    sample_reports = brca_report.iloc[:500, :]


    base_schema = BaseResponse.model_json_schema()
    full_schema = Response.model_json_schema()

    ##### baseline #####
    print("baseline")
    correct_count = 0
    incorrect_count = 0
    parsing_error = 0
    start_time = time.time()
    for idx, row in sample_reports.iterrows():
        print(idx)
        report = row["text"]
        label = row["t"]
        prompt = baseline_prompt.format(report=row["text"])
        data = get_response(client, idx, system_instruction+"\n"+prompt, base_schema)
        if not data:
            parsing_error += 1
            continue
        if f"T{label+1}" == data["predictedStage"]:
            result = "Correct!"
            correct_count += 1
        else:
            result = "Wrong!"
            incorrect_count +=1
        lines = [
            f"index: {idx}",
            result,
            f"label: T{label+1}",
            f"pred: {data['predictedStage']}",
            f"reasoning : {None}",
            f"rules : {None}"
        ]
        plot_in_box(lines)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    print(f"correct: {correct_count}, wrong: {incorrect_count}, parsing error: {parsing_error}")
    print("-"*20)

    ##### memory - always update #####
    print("memory - always update")
    memory = ""
    correct_count = 0
    incorrect_count = 0
    parsing_error = 0
    update_flag = True
    num_update = 0

    start_time = time.time()
    for idx, row in sample_reports.iterrows():
        print(idx)
        report = row["text"]
        label = row["t"]
        
        if memory == "":
            prompt = initial_predict_prompt.format(report=row["text"])
        else:
            prompt = subsequent_predict_prompt.format(memory=memory, report=row["text"])
            
        data = get_response(client, idx, system_instruction+"\n"+prompt, full_schema)
        if not data:
            parsing_error += 1
            continue
        print(f"similarity: {fuzz.ratio(memory, data['rules'])}")
        print(f"at {idx}, memory is updated")
        memory = data['rules']
        print(f"new memory: {memory}")
             
        if f"T{label+1}" == data["predictedStage"]:
            result = "Correct!"
            correct_count += 1
        else:
            result = "Wrong!"
            incorrect_count +=1
        lines = [
            f"index: {idx}",
            result,
            f"label: T{label+1}",
            f"pred: {data['predictedStage']}",
            f"reasoning : {data['reasoning']}",
            f"rules : {data['rules']}"
        ]
        plot_in_box(lines)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    print(f"correct: {correct_count}, wrong: {incorrect_count}, parsing error: {parsing_error}")
    print("-"*20)

    ##### memory - update only when sim(prev, curr) < threshold #####
    threshold = 90
    print(f"memory - update conditionally (threshold: {threshold})")
    memory = "" # a list of strings
    correct_count = 0
    incorrect_count = 0
    parsing_error = 0
    update_flag = False
    num_update = 0

    start_time = time.time()
    for idx, row in sample_reports.iterrows():
        print(idx)
        report = row["text"]
        label = row["t"]
        
        if memory == "":
            prompt = initial_predict_prompt.format(report=row["text"])
            data = get_response(client, idx, system_instruction+"\n"+prompt, full_schema)
            if not data:
                parsing_error += 1
                continue
            memory = data['rules']
            print(f"Initial memory: {memory}\n")
      
        else:
            prompt = subsequent_predict_prompt.format(memory=memory, report=row["text"])
            data = get_response(client, idx, system_instruction+"\n"+prompt, full_schema)
            if not data:
                parsing_error += 1
                continue
            print(f"similarity: {fuzz.ratio(memory, data['rules'])}")
            update_flag = is_updated(memory, data['rules'], threshold)
            if update_flag:
                num_update += 1
                print(f"at {idx}, memory is updated")
                memory = data['rules']
                print(f"new memory: {memory}")

        if f"T{label+1}" == data["predictedStage"]:
            result = "Correct!"
            correct_count += 1
        else:
            result = "Wrong!"
            incorrect_count +=1
        lines = [
            f"index: {idx}",
            result,
            f"label: T{label+1}",
            f"pred: {data['predictedStage']}",
            f"reasoning : {data['reasoning']}",
            f"rules: {data['rules']}"
        ]
        plot_in_box(lines)
        update_flag = False
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    print(f"correct: {correct_count}, wrong: {incorrect_count}, parsing error: {parsing_error}")
    print(f"memory is updated {num_update} times")
