
from transformers import AutoTokenizer
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/secure/chiahsuan/hf_cache/")

from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser

import pandas as pd

import requests

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def do_inference(data):
    """ send formatted data to llm
    """
    headers = {
    "Content-Type": "application/json",
    }

    response = requests.post(
        'http://127.0.0.1:8080/generate',
        headers=headers,
        json=data
    )
    
    return response.json()

class Response(BaseModel):
    predictedStage: str = Field(description="predicted cancer stage")
    reasoning: str = Field(description="reasoning to support predicted cancer stage")
    rules: List[str] = Field(description="list of rules")

parser = PydanticOutputParser(pydantic_object=Response)

initial_predict_prompt = """You are an expert at interpreting pathology reports for cancer staging. You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

What is the T stage from this report? Ignore any substaging information. Please select from the following four options:  T1, T2, T3, T4.
What is your reasoning to support your stage prediction?
Please induce a list of rules as knowledge that help you predict the next report. Make sure every rule does not contain any report-specific information. Instead, list general guidelines that apply to the specific cancer type and the AJCC staging system.

{format_instruction}"""

subsequent_predict_prompt = """You are an expert at interpreting pathology reports for cancer staging. You are provided with a pathology report for a cancer patient.
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

{format_instruction}"""

def is_updated(old_memory, new_memory, threshold):
    old_str = "\n".join(old_memory)
    new_str = "\n".join(new_memory)
    if fuzz.ratio(old_str, new_str) >= threshold : 
        return True # update memory
    else:
        return False

brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
sample_reports = brca_report.sample(n=500, random_state=123)


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

for threshold in range(100, -1, -10):
    memory = "" # a list of strings
    correct_count = 0
    incorrect_count = 0
    parsing_error_count = 0

    for idx, row in sample_reports.iterrows():
        report = row["text"]
        label = row["t"]
        
        if memory == "":
            chat = [
                {"role": "user", "content": initial_predict_prompt.format(report=report, format_instruction=parser.get_format_instructions())}
            ]

            templated_chat = tokenizer.apply_chat_template(chat, tokenize=False)
            
            data = {
                "inputs": templated_chat,
                "parameters": {
                    "do_sample": False,
                    "max_new_tokens": 1024
                }
            }

            response = do_inference(data)

            try:
                obj = parser.invoke(response['generated_text'])
            except:
                print(f"At {idx}, skip due to parsing error")
                parsing_error_count += 1
                continue
            
            memory = obj.rules
            print(f"Initial memory: {memory}\n")

            if f"T{label+1}" == obj.predictedStage:
                result = "Correct prediction"
                correct_count += 1
            else:
                result = f"Wrong prediction\nReasoning: {obj.reasoning}"
                incorrect_count += 1
            lines = [
                f"Report Index: {idx}",
                f"Label: T{label+1}",
                f"Prediction: {obj.predictedStage}",
                result
            ]
            plot_in_box(lines)

        else:
            sub_chat = [
                {"role": "user", "content": subsequent_predict_prompt.format(memory=memory, report=report, format_instruction=parser.get_format_instructions())}
            ]

            sub_templated_chat = tokenizer.apply_chat_template(sub_chat, tokenize=False)
            sub_data = {
                "inputs": sub_templated_chat,
                "parameters": {
                    "do_sample": False,
                    "max_new_tokens": 1024
                }
            }

            sub_response = do_inference(sub_data)
            
            try:
                sub_obj = parser.invoke(sub_response['generated_text'])
            except:
                print(f"At {idx}, skip due to parsing error")
                parsing_error_count += 1
                continue

            if is_updated(memory,sub_obj.rules, threshold):
                print(f"At {idx}, memory is updated")
                memory = sub_obj.rules
                print(f"New memory: {memory}")

            if f"T{label+1}" == sub_obj.predictedStage:
                result = "Correct prediction"
                correct_count += 1
            else:
                result = f"Wrong prediction\nReasoning: {sub_obj.reasoning}"
                incorrect_count += 1
            lines = [
                f"Report Index: {idx}",
                f"Label: T{label+1}",
                f"Prediction: {sub_obj.predictedStage}",
                result
            ]
            plot_in_box(lines)

    print(f"when threshold is {threshold}")
    print(f"correct: {correct_count}, incorrect: {incorrect_count}, parsing error: {parsing_error_count}")


print("The end")
print(correct_count, incorrect_count, parsing_error_count)


for idx, report in sample_reports.iterrows():
    print(idx)


