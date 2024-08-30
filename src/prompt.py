system_instruction = "You are an expert at interpreting pathology reports for cancer staging."

initial_predict_prompt_t14 = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic T stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your T stage prediction?

What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

Please induce a list of rules as knowledge that help you predict the T stage for the next report. Ensure there is at least one rule for each T stage (T1, T2, T3, T4) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.
"""

subsequent_predict_prompt_t14 = """You are provided with a pathology report for a cancer patient.
Here is a list of rules you have learned to correctly predict the T stage information:
{memory}

Please review this report and determine the pathologic T stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your T stage prediction?

What is the T stage from this report? Ignore any substaging information. Please select from the following four options: T1, T2, T3, T4.

What is your updated list of rules that help you predict the T stage for the next report? You can either modify the original rules or add new ones. Ensure there is at least one rule for each T stage (T1, T2, T3, T4) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.
"""

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

initial_predict_prompt_n03 = """You are provided with a pathology report for a cancer patient.
Please review this report and determine the pathologic N stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your N stage prediction?

What is the N stage from this report? Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

Please induce a list of rules as knowledge that help you predict the N stage for the next report. Ensure there is at least one rule for each N stage (N0, N1, N2, N3) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.
"""

subsequent_predict_prompt_n03 = """You are provided with a pathology report for a cancer patient.
Here is a list of rules you have learned to correctly predict the N stage information:
{memory}

Please review this report and determine the pathologic N stage of the patient's cancer.

Here is the report:
{report}

What is your reasoning to support your N stage prediction?

What is the N stage from this report? Ignore any substaging information. Please select from the following four options: N0, N1, N2, N3.

What is your updated list of rules that help you predict the N stage for the next report? You can either modify the original rules or add new ones. Ensure there is at least one rule for each N stage (N0, N1, N2, N3) in the list of rules. Ensure each rule is general and applicable to the specific cancer type and the AJCC's TNM Staging System, avoiding any report-specific information.   
"""

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

prompt_template='''
<|system|>:{system_instruction}
<|prompter|>:{prompt}
<|assistant|>:
'''

verification_prompt_t14 = """You are provided with the reasoning of another agent regarding the predicted T stage of breast cancer according to the AJCC's TNM Staging System. Your task is to evaluate this reasoning, focusing on the accuracy of the tumor size assessment and whether the correct T stage was assigned. Substaging is not required.

Below are some examples to guide your evaluation:

Reasoning: The report indicates that the patient has 'Infiltrating ductal carcinoma, Nottingham grade II (of III)' and the size of the mass is 'forming a 1.3 x 1.1 x 1.0 cm mass'. Based on the TNM staging system for breast cancer, a T stage is determined by the size of the primary tumor. T1 indicates a tumor that is 2 cm or less in size. Therefore, the T stage for this patient is T1.
Evaluation: The tumor size is given as 1.3 x 1.1 x 1.0 cm, with the maximum dimension being 1.3 cm. The reasoning correctly identifies that this size falls within the T1 category (2 cm or less). The arithmetic comparison is accurate, and the conclusion that the T stage is T1 is correct.

Reasoning: The T stage for cancer is determined by the size of the tumor. In this report, it is mentioned that the size of the tumor is 1.9 x 1.2 x 1.0 cm. According to the American Joint Committee on Cancer (AJCC) staging system, a T2 stage tumor is defined as a tumor with a size of 2.0 to 5.0 cm. Therefore, the T stage for this patient's cancer is T2.
Evaluation: The tumor size is given as 1.9 x 1.2 x 1.0 cm. The reasoning incorrectly concludes that the T stage is T2, stating that T2 corresponds to tumors between 2.0 and 5.0 cm. However, the actual tumor size (1.9 cm) falls within the T1 category, as it is less than 2 cm. The correct conclusion should be T1, not T2. It is important to double-check the boundary values when making such classifications.

Reasoning: The T stage of a cancer is determined by the size of the tumor. In this report, the size of the tumor is given as 2.8 cm in greatest dimension. According to the TNM staging system for cancer, a T2 stage tumor is defined as a tumor that is greater than 2 cm in size but not greater than 5 cm in size. Therefore, based on the information provided in this report, the patient's cancer has a pathologic T stage of T2.
Evaluation: The tumor size of 2.8 cm is correctly identified as falling within the T2 stage, defined as greater than 2 cm but not more than 5 cm. The arithmetic comparison is accurate, and the conclusion that the T stage is T2 is correct.

Reasoning: The tumor measures 3.2 x 2.5 x 1.8 cm. Based on the TNM staging system, a tumor of this size falls under the T3 category, as it is larger than 2 cm. Therefore, the T stage for this patient is T3.
Evaluation: The reasoning incorrectly assigns the T3 stage. A tumor with a greatest dimension of 3.2 cm falls within the T2 category (greater than 2 cm but not more than 5 cm). The correct conclusion should be T2, not T3. It is crucial to recheck the TNM staging criteria to ensure the tumor size matches the correct stage.

Reasoning: The pathology report indicates that the tumor size is 6.5 cm in its greatest dimension. According to the TNM staging system, a T3 stage tumor is defined as one that is greater than 5 cm in size. Therefore, the T stage for this patient is T3.
Evaluation: The tumor size of 6.5 cm is correctly categorized as T3, which includes tumors greater than 5 cm. The arithmetic comparison is accurate, and the conclusion that the T stage is T3 is correct.

Reasoning: The report indicates that the tumor measures 70 mm in greatest dimension (7 x 3 x 4.5 cm). According to the TNM staging system for cancer, a T2 stage is defined as a tumor that is 2 cm or more in size but not more than 5 cm. Therefore, the T stage for this patient's cancer is T2.
Evaluation: The tumor size is given as 70 mm (7 cm), which should correctly place it in the T3 category, as T3 includes tumors greater than 5 cm. The reasoning incorrectly assigns a T2 stage, which is inaccurate. The correct conclusion should be T3, not T2. Verify the unit conversions and arithmetic to ensure accuracy in staging.

Reasoning: The T stage for cancer is determined by the size of the tumor and whether it has invaded nearby structures. In this report, the patient has a primary tumor with a greatest dimension of 4 cm, which would typically correspond to a T2 stage. However, the report also indicates that the tumor invades the dermal skin, which is a sign of T4 stage. Therefore, the T stage for this patient is T4.
Evaluation: The reasoning correctly identifies the tumor size as 4 cm, which would typically correspond to T2. However, the report also mentions dermal skin invasion, which is indicative of T4. The reasoning accurately concludes that the T stage should be T4 due to the invasion into the skin, overriding the size-based classification.

Reasoning: The tumor size is given as 4.5 cm, and it has invaded the chest wall. However, since the size is less than 5 cm, it should be classified as T2. Therefore, the T stage for this patient is T2.
Evaluation: The reasoning incorrectly focuses on the tumor size instead of the invasion into the chest wall. The invasion into the chest wall qualifies the tumor as T4, regardless of size. The correct conclusion should be T4, not T2.

Now, evaluate the following reasoning:

Reasoning: {reasoning}
Evaluation:
"""


verification_prompt_n03 = """You are provided with the reasoning of another agent regarding the predicted N stage of breast cancer according to the AJCC's TNM Staging System. Your task is to evaluate this reasoning, focusing on the accuracy of lymph node involvement assessment and whether the correct N stage was assigned. Substaging is not required.

Below are some examples to guide your evaluation:

Reasoning: The report indicates that 16 lymph nodes were identified and all were negative for malignancy. Therefore, the pathologic N stage for this patient's cancer is N0.
Evaluation: The reasoning correctly identifies that all 16 lymph nodes were negative for malignancy. The comparison of the number of malignant lymph nodes (0) against the threshold (presence of malignancy in any node) is accurate. The conclusion that the N stage is N0 is correct.

Reasoning: The report states that all 15 axillary lymph nodes examined were negative for metastasis, but because the tumor is large and aggressive, the N stage is assigned as N1 to reflect the potential for lymph node involvement.
Evaluation: The reasoning incorrectly assigns an N1 stage based on the tumor's characteristics rather than the actual lymph node involvement. According to the TNM staging system, N0 is defined as no regional lymph node involvement, regardless of tumor size or aggression. The correct conclusion should be N0, not N1. Reassess the criteria to ensure lymph node involvement is the primary factor in staging.

Reasoning: The report indicates that there were 12 lymph nodes examined, and metastatic ductal adenocarcinoma was found in two of them (2/12). This corresponds to an N1 stage.
Evaluation: The reasoning accurately identifies that 2 out of 12 lymph nodes are positive for metastatic carcinoma, which corresponds to N1. The arithmetic comparison is correctly made, as the N1 stage includes 1-3 positive axillary lymph nodes. The conclusion that the N stage is N1 is correct.

Reasoning: The N stage for cancer is determined by the number of lymph nodes involved. In this report, metastatic adenocarcinoma is identified in 3 of 27 right axillary lymph nodes. Therefore, the N stage is N2.
Evaluation: The reasoning identifies 3 out of 27 right axillary lymph nodes as positive for metastatic adenocarcinoma and concludes an N2 stage. However, N1 corresponds to 1 to 3 positive axillary lymph nodes, and the reasoning inaccurately categorizes the stage. The correct conclusion should be N1, not N2. Ensure that the number of positive nodes is correctly categorized.

Reasoning: The patient's pathology report indicates that there are eight lymph nodes positive for tumor in the right axilla, with extensive extracapsular extension and deposits larger than 2 mm. This information supports the classification of N2 (metastasis in 4 to 9 axillary lymph nodes).
Evaluation: The reasoning indicates that eight lymph nodes are positive for the tumor, with extracapsular extension. N2 corresponds to metastasis in 4 to 9 axillary lymph nodes. The reasoning correctly identifies that eight positive nodes fall within this range. The arithmetic comparison is accurate, leading to the correct conclusion that the N stage is N2.

Reasoning: The report states that 4 out of 12 axillary lymph nodes are positive for metastasis, and this corresponds to an N1 stage.
Evaluation: N1 corresponds to 1 to 3 positive axillary lymph nodes, while 4 nodes should be classified as N2. The correct conclusion should be N2, not N1. Reevaluate the reasoning to ensure correct staging based on the number of involved nodes.

Reasoning: The report indicates that eighteen of eighteen lymph nodes were involved by metastatic carcinoma with extranodal spread. This corresponds to an N3 stage.
Evaluation: The reasoning states that all 18 lymph nodes were positive, which fits the N3 category (10 or more positive axillary lymph nodes). The arithmetic assessment is accurate, leading to the correct conclusion that the N stage is N3.

Reasoning: The report indicates that 10 out of 18 axillary lymph nodes are positive for metastatic carcinoma and this corresponds to an N2 stage.
Evaluation: The reasoning incorrectly assigns an N2 stage. N2 corresponds to 4 to 9 positive axillary lymph nodes, while 10 nodes should be classified as N3. The correct conclusion should be N3, not N2. Double-check the node count against the staging criteria to ensure accuracy.

Now, evaluate the following reasoning:

Reasoning: {reasoning}
Evaluation:
"""

# verification_prompt_n03 = """You are provided with another agent's reasoning for its predicted N cancer stage in breast cancer, according to the AJCC's TNM Staging System.

# Please evaluate this reasoning, offer constructive feedback, and determine the final N stage.

# Here is the reasoning:
# {reasoning}

# Please provide:

# 1. Your feedback on the given reasoning.

# 2. The final N stage you conclude based on your assessment (Select from the following options: N0, N1, N2, N3).
# """


# numerical_verification_prompt_t14 = """You are provided with another agent's reasoning for predicting the T stage of breast cancer according to the AJCC's TNM Staging System. This reasoning includes an extracted largest tumor size and involves comparing this size against the staging criteria.

# Your tasks are:

# 1. Evaluate the reasoning provided to determine if the numerical comparisons were performed correctly.
# 2. Carefully check for any errors in measurement unit conversions (e.g., mm to cm) and ensure that comparisons are made correctly.
# 3. Based on the reasoning alone, determine the correct T stage.

# Please note that you do not have access to the original pathological report from which the size information was extracted by the other agent, and you must base your judgment solely on the reasoning provided.

# Here is the reasoning:
# {reasoning}

# Please provide:

# 1. Feedback on the accuracy of the numerical comparison.
# 2. The correct T stage you conclude based on your evaluation (Select from the following options: T1, T2, T3, T4).
# """


# numerical_verification_prompt_n03 = """You are provided with another agent's reasoning for predicting the N stage of breast cancer according to the AJCC's TNM Staging System. This reasoning includes an assessment of the number of regional lymph nodes that contain cancer and involves comparing this number against the staging criteria.

# Your tasks are:

# 1. Evaluate the reasoning provided to determine if the numerical comparisons were performed correctly.
# 2. Carefully check for any errors in measurement unit conversions (e.g., mm to cm) and ensure that comparisons are made correctly.
# 3. Based on the reasoning alone, determine the correct N stage.

# Please note that you do not have access to the original pathological report from which the lymph node information was extracted by the other agent, and you must base your judgment solely on the reasoning provided.

# Here is the reasoning:
# {reasoning}

# Please provide:

# 1. Feedback on the accuracy of the numerical comparison.
# 2. The correct N stage you conclude based on your evaluation (Select from the following options: N0, N1, N2, N3).
# """



# verification_wm_prompt_t14 = """You are provided with another agent's reasoning for its predicted T cancer stage in breast cancer, according to the AJCC's TNM Staging System.

# Please evaluate this reasoning, offer constructive feedback, and determine the final T stage.


# Here is the reasoning:
# {reasoning}

# Here are some rules you can reference to predict the T stage:
# T1: If the tumor is invasive and the tumor size is 2 cm or less, then the T stage is T1.
# T2: If the tumor is invasive and the tumor size is more than 2 cm but not more than 5 cm, then the T stage is T2.
# T3: If the tumor is invasive and the tumor size is more than 5 cm or if the tumor is invasive lobular carcinoma with a size that cannot be accurately determined and is estimated to be more than 5 cm, then the T stage is T3.
# T4: If the tumor is invasive and has direct extension to the chest wall or skin, then the T stage is T4.

# Please provide:

# 1. Your feedback on the given reasoning.

# 2. The final T stage you conclude based on your assessment (Select from the following options: T1, T2, T3, T4).
# """

# verification_wm_prompt_n03 = """You are provided with another agent's reasoning for its predicted N cancer stage in breast cancer, according to the AJCC's TNM Staging System.

# Please evaluate this reasoning, offer constructive feedback, and determine the final N stage.

# Here is the reasoning:
# {reasoning}

# Here are some rules you can reference to predict the N stage:
# N0: No regional lymph node metastasis identified histologically.
# N1: Metastasis in 1-3 axillary lymph nodes or in internal mammary nodes with clinically negative axillary nodes.
# N2: Metastasis in 4-9 axillary lymph nodes, or in clinically detected internal mammary nodes in the absence of axillary metastasis.
# N3: Metastasis in 10 or more axillary lymph nodes, or in inframammary and/or ipsilateral supraclavicular lymph nodes.

# Please provide:

# 1. Your feedback on the given reasoning.

# 2. The final N stage you conclude based on your assessment (Select from the following options: N0, N1, N2, N3).
# """


# numerical_verification_wm_prompt_t14 = """You are provided with another agent's reasoning for predicting the T stage of breast cancer according to the AJCC's TNM Staging System. This reasoning includes an extracted largest tumor size and involves comparing this size against the staging criteria.

# Your tasks are:

# 1. Evaluate the reasoning provided to determine if the numerical comparisons were performed correctly.
# 2. Carefully check for any errors in measurement unit conversions (e.g., mm to cm) and ensure that comparisons are made correctly.
# 3. Based on the reasoning alone, determine the correct T stage.

# Please note that you do not have access to the original pathological report from which the size information was extracted by the other agent, and you must base your judgment solely on the reasoning provided.

# Here is the reasoning:
# {reasoning}

# Here are some rules you can reference to predict the T stage:
# T1: If the tumor is invasive and the tumor size is 2 cm or less, then the T stage is T1.
# T2: If the tumor is invasive and the tumor size is more than 2 cm but not more than 5 cm, then the T stage is T2.
# T3: If the tumor is invasive and the tumor size is more than 5 cm or if the tumor is invasive lobular carcinoma with a size that cannot be accurately determined and is estimated to be more than 5 cm, then the T stage is T3.
# T4: If the tumor is invasive and has direct extension to the chest wall or skin, then the T stage is T4.

# Please provide:

# 1. Feedback on the accuracy of the numerical comparison.
# 2. The correct T stage you conclude based on your evaluation (Select from the following options: T1, T2, T3, T4).
# """


# numerical_verification_wm_prompt_n03 = """You are provided with another agent's reasoning for predicting the N stage of breast cancer according to the AJCC's TNM Staging System. This reasoning includes an assessment of the number of regional lymph nodes that contain cancer and involves comparing this number against the staging criteria.

# Your tasks are:

# 1. Evaluate the reasoning provided to determine if the numerical comparisons were performed correctly.
# 2. Carefully check for any errors in measurement unit conversions (e.g., mm to cm) and ensure that comparisons are made correctly.
# 3. Based on the reasoning alone, determine the correct N stage.

# Please note that you do not have access to the original pathological report from which the lymph node information was extracted by the other agent, and you must base your judgment solely on the reasoning provided.

# Here is the reasoning:
# {reasoning}

# Here are some rules you can reference to predict the N stage:
# N0: No regional lymph node metastasis identified histologically.
# N1: Metastasis in 1-3 axillary lymph nodes or in internal mammary nodes with clinically negative axillary nodes.
# N2: Metastasis in 4-9 axillary lymph nodes, or in clinically detected internal mammary nodes in the absence of axillary metastasis.
# N3: Metastasis in 10 or more axillary lymph nodes, or in inframammary and/or ipsilateral supraclavicular lymph nodes.

# Please provide:

# 1. Feedback on the accuracy of the numerical comparison.
# 2. The correct N stage you conclude based on your evaluation (Select from the following options: N0, N1, N2, N3).
# """