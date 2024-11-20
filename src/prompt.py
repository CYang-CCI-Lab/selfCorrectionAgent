system_instruction = (
    "You are an expert at interpreting pathology reports for cancer staging."
)

# initial_predict_prompt_t14 = """You are provided with a pathology report for a cancer patient.
# Please review this report and determine the pathologic T stage of the patient's cancer.

# Here is the report:
# {report}

# What is your reasoning to support your T stage prediction?

# What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

# Please induce a list of rules as knowledge that help you predict the T stage for the next report. Ensure there is at least one rule for each T stage (T1, T2, T3, T4) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.
# """

# subsequent_predict_prompt_t14 = """You are provided with a pathology report for a cancer patient.
# Here is a list of rules you have learned to correctly predict the T stage information:
# {memory}

# Please review this report and determine the pathologic T stage of the patient's cancer.

# Here is the report:
# {report}

# What is your reasoning to support your T stage prediction?

# What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

# What is your updated list of rules that help you predict the T stage for the next report? You can either modify the original rules or add new ones. Ensure there is at least one rule for each T stage (T1, T2, T3, T4) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.
# """

testing_predict_prompt_t14 = """You are provided with a pathology report for a cancer patient.
Here is a list of rules you have learned to correctly predict the T stage information:
{memory}

Please review this report and determine the pathologic T stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your T stage prediction? 

What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

"""

zs_predict_prompt_t14 = """You are provided with a pathology report for a cancer patient.

Please review this report and determine the pathologic T stage of the patient's cancer.

Here is the report:
{report}

What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

"""

zscot_predict_prompt_t14 = """You are provided with a pathology report for a cancer patient.

Please review this report and determine the pathologic T stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your T stage prediction? 

What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

"""

# initial_predict_prompt_n03 = """You are provided with a pathology report for a cancer patient.
# Please review this report and determine the pathologic N stage of the patient's cancer.

# Here is the report:
# {report}

# What is your reasoning to support your N stage prediction?

# What is the N stage from this report? Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

# Please induce a list of rules as knowledge that help you predict the N stage for the next report. Ensure there is at least one rule for each N stage (N0, N1, N2, N3) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.
# """

# subsequent_predict_prompt_n03 = """You are provided with a pathology report for a cancer patient.
# Here is a list of rules you have learned to correctly predict the N stage information:
# {memory}

# Please review this report and determine the pathologic N stage of the patient's cancer.

# Here is the report:
# {report}

# What is your reasoning to support your N stage prediction?

# What is the N stage from this report? Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

# What is your updated list of rules that help you predict the N stage for the next report? You can either modify the original rules or add new ones. Ensure there is at least one rule for each N stage (N0, N1, N2, N3) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.   
# """

testing_predict_prompt_n03 = """You are provided with a pathology report for a cancer patient.
Here is a list of rules you have learned to correctly predict the N stage information:
{memory}

Please review this report and determine the pathologic N stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your N stage prediction?

What is the N stage from this report? Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

"""

zs_predict_prompt_n03 = """You are provided with a pathology report for a cancer patient.

Please review this report and determine the pathologic N stage of the patient's cancer.

Here is the report:
{report}

What is the N stage from this report? Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

"""

zscot_predict_prompt_n03 = """You are provided with a pathology report for a cancer patient.

Please review this report and determine the pathologic N stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your N stage prediction?

What is the N stage from this report? Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

"""

prompt_template = """
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
"""
