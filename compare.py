import os
import json
import pandas as pd
import random
from typing import List, Dict, Optional, Literal
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI
import copy

# ----------------------------------------------------------------------
# System instruction & Prompts
# ----------------------------------------------------------------------

system_instruction = "You are an expert in determining the N stage of breast cancer following the AJCC Cancer Staging Manual (7th edition)."

# Prompt that asks model to create rules from its own memory (LLM-based)
prompt_generate_llm_rules = """
Please articulate the N staging rules for breast cancer, focusing only on N0, N1, N2, and N3.
Ignore any sub-stages (e.g., N1mi, N1b).
Return your answer as a list of short statements that align strictly with the AJCC Cancer Staging Manual (7th edition).
"""

# Prompt that asks model to create rules from a RAG-based excerpt
prompt_generate_rag_rules = """
You are provided with an excerpt from the AJCC Cancer Staging Manual (7th edition) for breast cancer, focusing on N stage classification.

Excerpt:
{rag_context}

Please:
1. Summarize the guidelines for N0, N1, N2, and N3 (no sub-stages) in bullet form.
2. Return your final summary as a list of short statements that align strictly with the excerpt.
"""

# Zero-shot test
prompt_test_zero_shot = """
You are provided with a pathology report of a breast cancer patient.

Pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the N stage.
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

# Memory-only test
prompt_test_memory_only = """
You are provided with a set of memory items (lessons learned from past mistakes or borderline cases) and a pathology report for a breast cancer patient.

Memory items:
{all_memory}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step.
   - Use the memory items to address any subtle or borderline issues.
   - If a memory item conflicts with the AJCC Cancer Staging Manual (7th edition), note the conflict and favor the AJCC Cancer Staging Manual (7th edition).
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

# Rules-only (used for both LLM-based or RAG-based rules)
prompt_test_with_base_rules = """
You are provided with official N staging rules for breast cancer (N0, N1, N2, N3), derived from the AJCC Cancer Staging Manual (7th edition), 
and a pathology report for a breast cancer patient.

AJCC N staging rules (7th edition):
{base_rules}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step, strictly applying these AJCC 7th edition rules.
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

# Rules + memory (used for both LLM-based or RAG-based rules)
prompt_test_with_rules_and_memory = """
You are provided with official N staging rules for breast cancer (N0, N1, N2, N3), derived from the AJCC Cancer Staging Manual (7th edition), 
a memory of lessons learned from past mistakes, and a pathology report.

AJCC N staging rules (7th edition):
{base_rules}

Memory items:
{all_memory}

Pathology report:
{report}

Please:
1. Combine these AJCC 7th edition rules and the memory items to determine the N stage.
   - The official AJCC Cancer Staging Manual (7th edition) has priority, but use memory items for subtle or borderline aspects.
   - If any memory item conflicts with the AJCC 7th edition rules, ignore or adjust it in favor of those rules.
2. Explain your reasoning step by step.
3. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

# RAG-only (excerpt-only)
prompt_test_rag = """
You are provided with relevant context retrieved from the AJCC Cancer Staging Manual (7th edition) for breast cancer (N0, N1, N2, N3),
and a pathology report for a breast cancer patient.

Excerpt from AJCC Cancer Staging Manual (7th edition):
{rag_context}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step, integrating this AJCC 7th edition excerpt.
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

# Memory creation
prompt_memory_creation = """
Your previous reasoning and predicted N stage are:
- Reasoning: {model_reasoning}
- Predicted N stage: {model_predicted_stage}

However, the actual ground truth N stage is: {ground_truth_stage}.

Below are the AJCC rules (7th edition):
{base_rules}

Please do the following:

1. Check your reasoning against the AJCC 7th edition rules and the ground truth.
   - Identify any point where your reasoning might diverge from these rules or from the ground truth.

2A. If you believe the ground truth is correct (or see no compelling reason to doubt it):
   - Create your memory with a concise lesson learned, clarifying how to correctly handle a similar scenario in the future.
   - Ensure your lesson does NOT conflict with the AJCC Cancer Staging Manual (7th edition).
   - No unverified or contradictory heuristics allowed.

2B. If you believe the ground truth is incorrect (i.e., it clearly contradicts the AJCC 7th edition rules):
   - Return an empty string.
"""

# Memory refinement
prompt_memory_refinement = """
Your current memory (accumulated lessons) is shown below:
{all_accumulated_memory}

Below are the official AJCC N staging rules (7th edition):
{base_rules}

Please refine your memory by:
1. Removing any duplications or contradictions with these AJCC 7th edition rules.
2. Keeping only additional insights or clarifications that address common pitfalls or borderline cases (i.e., those not explicitly clarified by AJCC).
3. Returning the cleaned-up memory as a concise set of bullet points or short statements.
"""

# ----------------------------------------------------------------------
# Example Pydantic Models & Agent
# (You can reuse your existing code logic, just with the updated prompts)
# ----------------------------------------------------------------------

class ResponseRules(BaseModel):
    rules: List[str] = Field(
        description="N staging rules (N0, N1, N2, N3) for breast cancer, from the AJCC Cancer Staging Manual (7th edition)."
    )

class ResponseStage(BaseModel):
    reasoning: str = Field(
        description="A step-by-step explanation of how the model arrived at the predicted N stage."
    )
    stage: Literal["N0", "N1", "N2", "N3"] = Field(
        description="The final predicted N stage, one of N0, N1, N2, or N3."
    )

class ResponseMemoryCreation(BaseModel):
    memory: str = Field(
        description="A concise string capturing the lesson(s) learned from a mismatch."
    )

class ResponseMemoryRefinement(BaseModel):
    refined_memory: List[str] = Field(
        description="A cleaned-up set of memory items after removing duplications or contradictions."
    )

schema_set = {
    "rule_generation": ResponseRules.model_json_schema(),
    "stage_prediction": ResponseStage.model_json_schema(),
    "memory_creation": ResponseMemoryCreation.model_json_schema(),
    "memory_refinement": ResponseMemoryRefinement.model_json_schema()
}


class Agent:
    def __init__(self, client: OpenAI, model: str, label: str, test_name: str):
        self.client = client
        self.model = model
        self.label = label
        self.test_name = test_name
        self.all_memory: List[str] = []
        # etc. (You can keep your existing code structure, referencing the updated prompts above)
        # ...

    # ... (Your existing methods, referencing prompt_generate_llm_rules, prompt_generate_rag_rules, etc.)

    # run_experiment and any other methods you already have...
