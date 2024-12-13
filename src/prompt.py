prompt_template_med42 = """
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
"""
system_instruction = "You are an expert in determining the T stage of breast cancer following the AJCC TNM classification system."


prompt_zscot_t14 = """Here is the pathology report:
{report}

Please:
1. Explain your reasoning step-by-step as you determine the T stage.
2. Provide your predicted T stage. Choose from T1, T2, T3, T4.
"""


prompt_rule_extraction_t14 = """Your previous reasoning and predicted T stage are:
- Reasoning: {model_reasoning}
- Your Predicted T stage: {model_predicted_stage}

The actual ground truth T stage for this report is: {ground_truth_T_stage}.

Please:
1. Identify where your reasoning diverged from the correct reasoning that leads to the ground truth.
2. Explain what key criteria or guidelines should have been applied differently.
3. Based on this analysis, induce several general rules that would help correctly identify the T stage in a similar scenario next time. 
   - These rules should be general and not tied to this specific patient's report.
   - They should clearly relate to accepted AJCC criteria.
   - Provide at least one or two rules that directly address the error.

"""

prompt_rule_refinement_t14 = """Below is a large set of rules accumulated from analyzing pathology reports. These rules were created iteratively and may include redundancies, overly specific guidance, or conflicting points.

Your task:
1. Review and consolidate the entire rule set into a smaller, cleaner, and more authoritative list of rules for determining T stages in breast cancer according to AJCC TNM guidelines.
2. Remove any redundant or contradictory rules.
3. Ensure each stage (T1, T2, T3, T4) is clearly defined with proper criteria.
4. Avoid rules that rely on case-specific details; keep them general and guideline-based.
5. The final set of rules should be logically organized, easy to follow, and helpful for future classification tasks.

Here is the accumulated rule set:

{all_accumulated_rules}

"""


prompt_test_with_rules_t14 = """You are provided with a pathology report for a breast cancer patient and a list of rules for determining the T stage.
Please review this report and determine the pathologic T stage of the patient's breast cancer, with the help of the rules.

Here are the rules for determining the T stage:
{context}

Here is the pathology report:
{report}

Please:
1. Explain your reasoning step-by-step as you determine the T stage.
2. Provide your predicted T stage, choosing from T1, T2, T3, T4.
"""