# prompt_template_med42 = """
# <|system|>:{system_instruction}
# <|prompter|>:{prompt}
# <|assistant|>:
# """
# system_instruction = "You are an expert in determining the T stage of breast cancer following the AJCC TNM classification system."


# prompt_cold_start = """Here is the pathology report:
# {report}

# Please:
# 1. Explain your reasoning step-by-step as you determine the T stage.
# 2. Provide your predicted T stage. Choose from T1, T2, T3, T4.
# """


# prompt_rule_extraction = """Your previous reasoning and predicted T stage are:
# - Reasoning: {model_reasoning}
# - Your Predicted T stage: {model_predicted_stage}

# The actual ground truth T stage for this report is: {ground_truth_T_stage}.

# Please:
# 1. Identify where your reasoning diverged from the correct reasoning that leads to the ground truth.
# 2. Explain what key criteria or guidelines should have been applied differently.
# 3. Based on this analysis, induce several general rules that would help correctly identify the T stage in a similar scenario next time. 
#    - These rules should be general and not tied to this specific patient's report.
#    - They should clearly relate to accepted AJCC criteria.
#    - Provide at least one or two rules that directly address the error.

# """

# prompt_rule_refinement = """Below is a large set of rules accumulated from analyzing pathology reports. These rules were created iteratively and may include redundancies, overly specific guidance, or conflicting points.

# Your task:
# 1. Review and consolidate the entire rule set into a smaller, cleaner, and more authoritative list of rules for determining T stages in breast cancer according to AJCC TNM guidelines.
# 2. Remove any redundant or contradictory rules.
# 3. Ensure each stage (T1, T2, T3, T4) is clearly defined with proper criteria.
# 4. Avoid rules that rely on case-specific details; keep them general and guideline-based.
# 5. The final set of rules should be logically organized, easy to follow, and helpful for future classification tasks.

# Here is the accumulated rule set:

# {all_accumulated_rules}

# """


# prompt_test_with_rules = """You are provided with a pathology report for a breast cancer patient and a list of rules for determining the T stage.
# Please review this report and determine the pathologic T stage of the patient's breast cancer, with the help of the rules.

# Here are the rules for determining the T stage:
# {context}

# Here is the pathology report:
# {report}

# Please:
# 1. Explain your reasoning step-by-step as you determine the T stage.
# 2. Provide your predicted T stage, choosing from T1, T2, T3, T4.
# """

system_instruction = "You are an expert in determining the T stage of breast cancer following the AJCC TNM classification system."

prompt_cold_start = """Here is the pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the T stage.
2. Provide your predicted T stage. Choose from T1, T2, T3, T4.
"""

prompt_memory_creation = """Your previous reasoning and predicted T stage are:
- Reasoning: {model_reasoning}
- Your Predicted T stage: {model_predicted_stage}

The actual ground truth T stage for this report is: {ground_truth_T_stage}.

Please:
1. Identify where your reasoning diverged from the correct reasoning that leads to the ground truth and which key AJCC guidelines or criteria should have been applied differently.
2. Based on 1, create your “memory” to ensure more accurate T-stage determinations in similar future cases. This memory should be general and not tied to this specific patient.
"""

# The accumulated memory items are saved in a list
# If len(all_accumulated_memory) >= 10 or any(len(item) >= 300 for item in all_accumulated_memory), the memory refinement task will be triggered
prompt_memory_refinement = """
Below is a large set of memory items accumulated from analyzing various pathology reports. These memory entries were created iteratively and may include redundancies, overly specific guidance, or conflicting points.

Your task is to review and consolidate this entire memory set into a smaller, cleaner, and more authoritative memory for determining T stages in breast cancer according to AJCC TNM guidelines by:
1. Removing any redundant or contradictory memory entries.
2. Avoiding memory points that rely on overly case-specific details; keep them general and guideline-based.

The final memory should be logically organized, easy to follow, and helpful for future classification tasks.

Here is the accumulated memory:

{all_accumulated_memory}
"""

# If len(all_accumulated_memory) > 0 and len(all_accumulated_memory) < 10 and all(len(item) < 300 for item in all_accumulated_memory), the memory test task will be triggered
prompt_test_with_memory = """You are provided with a pathology report for a breast cancer patient and a set of memory items that guide T stage determination.

Here is the relevant memory:
{context}

Here is the pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the T stage.
2. Provide your predicted T stage, choosing from T1, T2, T3, T4.
"""

