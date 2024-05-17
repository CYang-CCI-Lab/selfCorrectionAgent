import pandas as pd
from openai import OpenAI
import openai
from tqdm import tqdm
from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field
from typing import List
import json
import time

openai_api_key = "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"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
brca_half1 = brca_report.iloc[:500, :]
# brca_half2 = brca_report.iloc[500:, :]
sample_reports = brca_half1 # brca_report.sample(n=50, random_state=123)

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

class BaseResponse(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")

class Response(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage") # 3
    reasoning: str = Field(description="reasoning to support predicted cancer stage") # 2
    rules: List[str] = Field(description="list of rules") # 1

based_required_json_schema = BaseResponse.model_json_schema()
required_json_schema = Response.model_json_schema()

system_instruction = "You are an expert at interpreting pathology reports for cancer staging."

baseline_prompt = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

What is the T stage from this report? Ignore any substaging information. Please select from the following four options:  T1, T2, T3, T4.
"""

initial_predict_prompt = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

What is the T stage from this report? Ignore any substaging information. Please select from the following four options:  T1, T2, T3, T4.
What is your reasoning to support your stage prediction?
Please induce a list of rules as knowledge that help you predict the next report. Make sure every rule does not contain any report-specific information. Instead, list general guidelines that apply to the specific cancer type and the AJCC staging system.
"""

subsequent_predict_prompt = """You are provided with a pathology report for a cancer patient.
Here is a list of rules you leanred to correctly predict the cancer stage information:
```
{memory}
```

Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

What is the T stage from this report? Ignore any substaging information. Please select from the following four options:  T1, T2, T3, T4.
What is your reasoning to support your stage prediction?
What is your updated list of rules that help you predict the next report? You can either modify the original rules or add new rules. Make sure every rule does not contain any report-specific information. Instead, list general guidelines that apply to the specific cancer type and the AJCC staging system.
"""

prompt_template='''
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
'''

# baseline
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
    # response = client.chat.completions.create(
    #         model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #         messages=[{"role":"system", "content": system_instruction},
    #                     {"role": "user", "content": prompt}],
    #         extra_body={"guided_json":based_required_json_schema},
    #         temperature = 0.0)
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": "[INST] "+system_instruction+"\n"+prompt+" [/INST]"}],
        extra_body={"guided_json":based_required_json_schema},
        temperature = 0.0)
    try:
        data = json.loads(response.choices[0].message.content.replace("\\", "\\\\"))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON at {idx}")
        print(response.choices[0].message.content)
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
        f"pred: {data['predictedStage']}"
    ]
    plot_in_box(lines)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
print(f"correct: {correct_count}, wrong: {incorrect_count}, parsing error: {parsing_error}")
print("-"*20)
print("memory - always update")
# memory - always update
memory = "" # a list of strings
correct_count = 0
incorrect_count = 0
parsing_error = 0
similarity = 0

start_time = time.time()
for idx, row in sample_reports.iterrows():
    print(idx)
    report = row["text"]
    label = row["t"]
    
    if memory == "":
        prompt = initial_predict_prompt.format(report=row["text"])
        # response = client.chat.completions.create(
        #         model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        #         messages=[{"role":"system", "content": system_instruction},
        #                   {"role": "user", "content": prompt}],
        #         extra_body={"guided_json":required_json_schema},
        #         temperature = 0.0)
        response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": system_instruction+"\n"+prompt}],
        extra_body={"guided_json":required_json_schema},
        temperature = 0.0)
        try:
            data = json.loads(response.choices[0].message.content.replace("\\", "\\\\"))
            memory = data['rules']
    
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at {idx}")
            print(response.choices[0].message.content)
            parsing_error += 1
            continue
    else:
        prompt = subsequent_predict_prompt.format(memory=memory, report=row["text"])
        
        # response = client.chat.completions.create(
        #         model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        #         messages=[{"role":"system", "content": system_instruction},
        #                   {"role": "user", "content": prompt}],
        #         extra_body={"guided_json":required_json_schema},
        #         temperature = 0.0)
        response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": system_instruction+"\n"+prompt}],
        extra_body={"guided_json":required_json_schema},
        temperature = 0.0)

        try:
            data = json.loads(response.choices[0].message.content.replace("\\", "\\\\"))
            similarity = fuzz.ratio(memory, data['rules'])
            memory = data['rules']
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at {idx}")
            print(response.choices[0].message.content)
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
        f"similarity: {similarity}",
        f"label: T{label+1}",
        f"pred: {data['predictedStage']}",
        f"reasoning : {data['reasoning']}",
        f"memory: {memory}"
    ]
    plot_in_box(lines)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")
print(f"correct: {correct_count}, wrong: {incorrect_count}, parsing error: {parsing_error}")
print("-"*20)
print("memory - update conditionally")
# memory - update only when sim(prev, curr) < threshold
threshold = 90
memory = "" # a list of strings
correct_count = 0
incorrect_count = 0
parsing_error = 0
update_flag = False
num_update = 0

start_time = time.time()
for idx, row in sample_reports.iterrows():
    if idx < 236:
        continue
    print(idx)
    report = row["text"]
    label = row["t"]
    
    if memory == "":
        prompt = initial_predict_prompt.format(report=row["text"])
        # response = client.chat.completions.create(
        #         model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        #         messages=[{"role":"system", "content": system_instruction},
        #                   {"role": "user", "content": prompt}],
        #         extra_body={"stop_token_ids":[128001,128009], 
        #                     "guided_json":required_json_schema},
        #         temperature = 0.0)
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{"role": "user", "content": system_instruction+"\n"+prompt}],
                extra_body={"guided_json":required_json_schema},
                temperature = 0.0)
            except openai.APITimeoutError as e:
                if attempt < 4:
                    wait_time = 2 * attempt 
                    print(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Request failed.")
                    raise e
          

        try:
            data = json.loads(response.choices[0].message.content.replace("\\", "\\\\"))
            memory = data['rules']
            print(f"Initial memory: {memory}\n")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at {idx}")
            print(response.choices[0].message.content)
            parsing_error += 1
            continue
    else:
        prompt = subsequent_predict_prompt.format(memory=memory, report=row["text"])
        
        # response = client.chat.completions.create(
        #         model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        #         messages=[{"role":"system", "content": system_instruction},
        #                   {"role": "user", "content": prompt}],
        #         extra_body={"stop_token_ids":[128001,128009],
        #                     "guided_json":required_json_schema},
        #         temperature = 0.0)
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{"role": "user", "content": system_instruction+"\n"+prompt}],
                extra_body={"guided_json":required_json_schema},
                temperature = 0.0)
                break
            except openai.APITimeoutError as e:
                if attempt < 4:
                    wait_time = 2 * attempt
                    print(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Request failed.")
                    raise e

        try:
            data = json.loads(response.choices[0].message.content.replace("\\", "\\\\"))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at {idx}")
            print(response.choices[0].message.content)
            parsing_error += 1
            continue
        if is_updated(memory, data['rules'], threshold):
            update_flag = True
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
        f"is updated?: {update_flag}"
    ]
    plot_in_box(lines)
    update_flag = False
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
print(f"correct: {correct_count}, wrong: {incorrect_count}, parsing error: {parsing_error}")
print(f"memory is updated {num_update} times")