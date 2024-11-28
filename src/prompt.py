prompt_template_med42 = """
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
"""
system_instruction = (
    "You are an expert at interpreting pathology reports for cancer staging."
)

####### Zero-shot Prompt #######
zs_t14 = """You are provided with a pathology report for a breast cancer patient.

Please review this report and determine the pathologic T stage of the patient's breast cancer based on the AJCC's TNM Staging System.

Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

Here is the report:
{report}
"""

zs_n03 = """You are provided with a pathology report for a breast cancer patient.

Please review this report and determine the pathologic N stage of the patient's breast cancer based on the AJCC's TNM Staging System.

Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

Here is the report:
{report}
"""


####### Zero-shot Chain of Thought Prompts #######
zscot_t14 = """You are provided with a pathology report for a breast cancer patient.

Please review this report and determine the pathologic T stage of the patient's breast cancer based on the AJCC's TNM Staging System.

As you analyze the report, explain step-by-step how you are interpreting the relevant information according to the AJCC guidelines and how it leads to the final decision regarding the T stage. Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

Here is the report:
{report}
"""

zscot_n03 = """You are provided with a pathology report for a breast cancer patient.

Please review this report and determine the pathologic N stage of the patient's breast cancer based on the AJCC's TNM Staging System.

As you analyze the report, explain step-by-step how you are interpreting the relevant information according to the AJCC guidelines and how it leads to the final decision regarding the N stage. Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

Here is the report:
{report}
"""

####### Retrieval-Augmented Generation Prompts #######
rag_t14 = """You are provided with a pathology report for a breast cancer patient and relevant chunks from the AJCC's TNM Staging System guidelines as context.

Please review this report and determine the pathologic T stage of the patient's breast cancer, with the help of the context.

As you analyze the report, explain step-by-step how you are interpreting the relevant information according to the AJCC guidelines and how it leads to the final decision regarding the T stage. Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

Here is the report:
{report}

Here is the context from the AJCC guidelines:
{context}
"""

rag_n03 = """You are provided with a pathology report for a breast cancer patient and relevant chunks from the AJCC's TNM Staging System guidelines as context.

Please review this report and determine the pathologic N stage of the patient's breast cancer, with the help of the context.

As you analyze the report, explain step-by-step how you are interpreting the relevant information according to the AJCC guidelines and how it leads to the final decision regarding the N stage. Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

Here is the report:
{report}

Here is the context from the AJCC guidelines:
{context}
"""

####### Long-term Memory Prompts #######
ltm_t14 = """You are provided with a pathology report for a breast cancer patient and a list of rules for determining the T stage.

Please review this report and determine the pathologic T stage of the patient's breast cancer, with the help of the rules.

As you analyze the report, explain step-by-step how you are interpreting the relevant information according to the rules and how it leads to the final decision regarding the T stage. Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

Here is the report:
{report}

Here are the rules for determining the T stage:
{context}
"""

ltm_n03 = """You are provided with a pathology report for a breast cancer patient and a list of rules for determining the N stage.

Please review this report and determine the pathologic N stage of the patient's breast cancer, with the help of the rules.

As you analyze the report, explain step-by-step how you are interpreting the relevant information according to the rules and how it leads to the final decision regarding the N stage. Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

Here is the report:
{report}

Here are the rules for determining the N stage:
{context}
"""
