system_instruction = "You are an expert at interpreting pathology reports for cancer staging."

baseline_prompt = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.
"""

initial_predict_prompt = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

1. What is your reasoning to support your stage prediction?
2. What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.
3. Please induce a list of rules as knowledge that help you predict the next report. Ensure each rule is general and applicable to the specific cancer type and the AJCC staging system, avoiding any report-specific information.
"""

subsequent_predict_prompt = """You are provided with a pathology report for a cancer patient.
Here is a list of rules you have learned to correctly predict the cancer stage information:
{memory}

Please review this report and determine the pathologic stage of the patient's cancer.

Here is the report:
```
{report}
```

1. What is your reasoning to support your stage prediction?
2. What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.
3. What is your updated list of rules that help you predict the next report? You can either modify the original rules or add new ones. Ensure each rule is general and applicable to the specific cancer type and the AJCC staging system, avoiding any report-specific information.
"""
