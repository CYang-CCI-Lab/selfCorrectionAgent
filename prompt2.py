# For N
system_instruction = "You are an expert in determining the N stage of breast cancer following the AJCC TNM classification system."

prompt_generate_rules = """
Please articulate the N staging rules for breast cancer, focusing only on N0, N1, N2, and N3.
Ignore any sub-stages (e.g., N1mi, N1b).
Return your answer as a list of short statements that align strictly with official AJCC 8th edition guidelines.
"""

prompt_test_zero_shot = """
You are provided with a pathology report of a breast cancer patient.

Pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the N stage.
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

prompt_test_memory_only = """
You are provided with a pathology report for a breast cancer patient and a set of memory items (lessons learned from past mistakes or borderline cases).

Memory items:
{all_memory}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step. 
   - Use the memory items to address any subtle or borderline issues.
   - If a memory item conflicts with AJCC guidelines, note the conflict and favor AJCC.
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

prompt_test_with_base_rules = """
You are provided with the official AJCC N staging rules for breast cancer (N0, N1, N2, N3) and a pathology report for a breast cancer patient.

AJCC N staging rules:
{base_rules}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step, strictly applying the AJCC rules.
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

prompt_memory_creation = """
Your previous reasoning and predicted N stage are:
- Reasoning: {model_reasoning}
- Predicted N stage: {model_predicted_stage}

However, the actual ground truth N stage is: {ground_truth_stage}.

Below are the AJCC rules:
{base_rules}

Please do the following:

1. Check your reasoning against the AJCC rules and the ground truth. 
   - Identify any point where your reasoning might diverge from these rules or from the ground truth.

2A. If you believe the ground truth is correct (or you see no compelling reason to doubt it):
   - Create your memory with a concise lesson learned, clarifying how to correctly handle a similar scenario in the future.
   - Ensure your lesson does NOT conflict with official AJCC guidelines (no unverified or contradictory heuristics).

2B. If you believe the ground truth is incorrect (i.e., it clearly contradicts the AJCC rules):
   - Return an empty string.
"""

prompt_memory_refinement = """
Your current memory (accumulated lessons) is shown below:
{all_accumulated_memory}

Below are the official AJCC N staging rules:
{base_rules}

Please refine your memory by:
1. Removing any duplications or contradictions with the AJCC rules.
2. Keeping only additional insights or clarifications that address common pitfalls or borderline cases (i.e., those not explicitly clarified by AJCC).
3. Returning the cleaned-up memory as a concise set of bullet points or short statements.
"""

prompt_test_with_rules_and_memory = """
You are provided with the official AJCC N staging rules for breast cancer (N0, N1, N2, N3), a memory of lessons learned from past mistakes, and a pathology report.

AJCC N staging rules:
{base_rules}

Memory items:
{all_memory}

Pathology report:
{report}

Please:
1. Combine the AJCC rules and the memory items to determine the N stage.
   - The official AJCC rules have priority, but use memory items for subtle or borderline aspects.
   - If any memory item conflicts with AJCC guidelines, ignore or adjust it.
2. Explain your reasoning step by step.
3. Provide your final predicted N stage (N0, N1, N2, or N3).