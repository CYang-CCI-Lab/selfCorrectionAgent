import os
import json
import pandas as pd
import random
from typing import List, Dict, Optional, Literal
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI
import copy

system_instruction = "You are an expert in determining the T stage of breast cancer following the AJCC (7th edition) TNM classification system."
prompt_generate_rules = """
Please articulate the T staging rules for breast cancer, focusing only on T1, T2, T3, and T4.
Ignore any sub-stages (e.g., T1a, T1b).
Return your answer as a list of short statements that align strictly with official AJCC 7th edition guidelines.
"""
prompt_test_zero_shot = """
You are provided with a pathology report of a breast cancer patient.

Pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the T stage.
2. Provide your final predicted T stage (T1, T2, T3, or T4).
"""

prompt_test_rag = """
You are provided with relevant context retrieved from the AJCC Cancer Staging Manual (7th edition) for breast cancer and a pathology report for a breast cancer patient.

Excerpt from AJCC Manual:
{rag_context}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step, integrating this AJCC excerpt.
2. Provide your final predicted T stage (T1, T2, T3, or T4).
"""

prompt_test_memory_only = """
You are provided with a set of memory items (lessons learned from past mistakes or borderline cases) and a pathology report for a breast cancer patient.

Memory items:
{all_memory}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step. 
   - Use the memory items to address any subtle or borderline issues.
   - If a memory item conflicts with AJCC guidelines, note the conflict and favor AJCC.
2. Provide your final predicted T stage (T1, T2, T3, or T4).
"""

# for the first report, or for baseline testing
prompt_test_with_base_rules = """
You are provided with the official AJCC T staging rules for breast cancer (T1, T2, T3, T4) and a pathology report for a breast cancer patient.

AJCC T staging rules:
{base_rules}

Pathology report:
{report}

Please:
1. Explain your reasoning step by step, strictly applying the AJCC rules.
2. Provide your final predicted T stage (T1, T2, T3, or T4).
"""

# if the model's prediction is wrong, do memory creation
prompt_memory_creation = """
Your previous reasoning and predicted T stage are:
- Reasoning: {model_reasoning}
- Predicted T stage: {model_predicted_stage}

However, the actual ground truth T stage is: {ground_truth_stage}.

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


# if the memory becomes too large or unwieldy

prompt_memory_refinement = """
Your current memory (accumulated lessons) is shown below:
{all_accumulated_memory}

Below are the official AJCC T staging rules:
{base_rules}

Please refine your memory by:
1. Removing any duplications or contradictions with the AJCC rules.
2. Keeping only additional insights or clarifications that address common pitfalls or borderline cases (i.e., those not explicitly clarified by AJCC).
3. Returning the cleaned-up memory as a concise set of bullet points or short statements.

"""


prompt_test_with_rules_and_memory = """
You are provided with the official AJCC T staging rules for breast cancer (T1, T2, T3, T4), a memory of lessons learned from past mistakes, and a pathology report.

AJCC T staging rules:
{base_rules}

Memory items:
{all_memory}

Pathology report:
{report}


Please:
1. Combine the AJCC rules and the memory items to determine the T stage.
   - The official AJCC rules have priority, but use memory items for subtle or borderline aspects.
   - If any memory item conflicts with AJCC guidelines, ignore or adjust it.
2. Explain your reasoning step by step.
3. Provide your final predicted T stage (T1, T2, T3, or T4).
"""


# ------------------------------------------------------
# Pydantic models (minimal changes from your examples)
# ------------------------------------------------------

class ResponseRules(BaseModel):
    rules: List[str] = Field(
        description="T staging rules (T1, T2, T3, T4) for breast cancer."
    )
class ResponseStage(BaseModel):
    reasoning: str = Field(
        description=(
            "A step-by-step explanation of how the model arrived at the predicted "
            "T stage, explicitly referencing numeric cutoffs or criteria."
        )
    )
    stage: Literal["T1", "T2", "T3", "T4"] = Field(
        description=(
            "The T stage determined from the report. Must be one of 'T1', 'T2', 'T3', or 'T4'. "
            "No additional sub-stages or variations allowed."
        )
    )

class ResponseMemoryCreation(BaseModel):
    memory: str = Field(
        description=(
            "A concise string capturing the lesson(s) learned from the mismatch. "
            "If the model believes the ground truth is incorrect, this can be an empty string."
        )
    )

class ResponseMemoryRefinement(BaseModel):
    refined_memory: List[str] = Field(
        description=(
            "A cleaner, more authoritative set of memory items after removing redundancies, "
            "contradictions, or confusion, without duplicating the base AJCC rules."
        )
    )


schema_set = {
    "rule_generation": ResponseRules.model_json_schema(),
    "stage_prediction": ResponseStage.model_json_schema(),
    "memory_creation": ResponseMemoryCreation.model_json_schema(),
    "memory_refinement": ResponseMemoryRefinement.model_json_schema()
}


class Agent:
    def __init__(
        self, 
        client: OpenAI, 
        model: str, 
        label: str, 
        test_name: str
    ) -> None:
        self.client = client
        self.model = model
        self.label = label
        self.test_name = test_name
        self.all_memory: List[str] = []
        self.base_rules: List[str] = []
        self.rag_context = self.get_rag_context()

    def get_rag_context(self) -> str:
        with open("context.json", "r") as f:
            context = json.load(f)
        context = context["rag_raw_t14"]
        return context
    
    def get_schema_followed_response(
        self, 
        messages: List[Dict], 
        schema: dict
    ) -> Optional[dict]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"guided_json": schema},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def run_experiment(
        self,
        dataset: pd.DataFrame,
        memory_threshold: int = 1,   # e.g., refine if we have more than {threshold} memory items
        max_memory_item_length: int = 1  # e.g., refine if any item is over {length} characters
    ) -> pd.DataFrame:

        parsing_error = 0
        memory_creation_indices = []  # to store row indices where memory creation happened

        # Get the rules for T staging
        print("Generating base rules...")
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_generate_rules}
        ]
        rules_output = self.get_schema_followed_response(messages, schema_set["rule_generation"])
        if rules_output:
            self.base_rules = rules_output["rules"]
        else:
            print("Error getting base rules.")
            return dataset
        
        print("Base rules acquired:", self.base_rules)
        
        dataset["current_memory"] = None
        dataset["current_memory"] = dataset["current_memory"].astype(object)

        pbar = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            report = row["text"]
            true_label = f"T{row[self.label]+1}"
            dataset.at[idx, "current_memory"] = copy.deepcopy(self.all_memory)  # store memory at each step

            # ----------------------------------------------
            #  Baseline #1: Zero-shot
            # ----------------------------------------------
            zero_shot_prompt = prompt_test_zero_shot.format(report=report)
            messages_zs = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": zero_shot_prompt}
            ]
            zs_response = self.get_schema_followed_response(messages_zs, schema_set["stage_prediction"])
            if zs_response:
                dataset.loc[idx, f"{self.test_name}_pred_zs"] = zs_response["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_zs"] = zs_response["reasoning"]

            # ---------------------------------------------
            #  Baseline #2: Rules-only
            # ---------------------------------------------
            rules_only_prompt = prompt_test_with_base_rules.format(
                report=report, 
                base_rules="\n".join(self.base_rules)
            )
            messages_rules = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": rules_only_prompt}
            ]
            rules_response = self.get_schema_followed_response(messages_rules, schema_set["stage_prediction"])
            if rules_response:
                dataset.loc[idx, f"{self.test_name}_pred_rules"] = rules_response["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_rules"] = rules_response["reasoning"]

            # ---------------------------------------------
            #  Baseline #3: Memory-only
            # ---------------------------------------------
            memory_context = "\n".join(self.all_memory) if self.all_memory else "No prior memory available."
            memory_only_prompt = prompt_test_memory_only.format(
                report=report, 
                all_memory=memory_context
            )
            messages_mem = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": memory_only_prompt}
            ]
            mem_response = self.get_schema_followed_response(messages_mem, schema_set["stage_prediction"])
            if mem_response:
                dataset.loc[idx, f"{self.test_name}_pred_memory_only"] = mem_response["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_memory_only"] = mem_response["reasoning"]

            # ---------------------------------------------
            #  Baseline #4: RAG 
            # ---------------------------------------------
            rag_prompt = prompt_test_rag.format(
                rag_context=self.rag_context, 
                report=report
            )
            messages_rag = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": rag_prompt}
            ]
            rag_response = self.get_schema_followed_response(messages_rag, schema_set["stage_prediction"])
            if rag_response:
                dataset.loc[idx, f"{self.test_name}_pred_rag"] = rag_response["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_rag"] = rag_response["reasoning"]

            # ---------------------------------------------
            #  Main approach: rules + memory
            # ---------------------------------------------
            # If no memory yet, we revert to rules-only prompt
            if not self.all_memory:
                filled_prompt = prompt_test_with_base_rules.format(report=report, base_rules="\n".join(self.base_rules))
            else:
                filled_prompt = prompt_test_with_rules_and_memory.format(
                    report=report, base_rules="\n".join(self.base_rules), all_memory="\n".join(self.all_memory)
                )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": filled_prompt}
            ]

            # Get response
            json_output = self.get_schema_followed_response(messages, schema_set["stage_prediction"])
            if not json_output:
                parsing_error += 1
                print(f"Error parsing JSON at row {idx}")
                dataset.loc[idx, f"{self.test_name}_parsed"] = False
                pbar.update(1)
                continue

            # Record model's reasoning & stage
            model_reasoning = json_output["reasoning"]
            model_predicted_stage = json_output["stage"]
            dataset.loc[idx, f"{self.test_name}_parsed"] = True
            dataset.loc[idx, f"{self.test_name}_reasoning"] = model_reasoning
            dataset.loc[idx, f"{self.test_name}_pred"] = model_predicted_stage

            # Check correctness
            if model_predicted_stage != true_label:
                # Trigger memory creation
                memory_creation_indices.append(idx)
                # Provide the model's output to the conversation
                messages.append({"role": "assistant", "content": str(json_output)})

                # Now build memory creation prompt
                creation_prompt = prompt_memory_creation.format(
                    model_reasoning=model_reasoning,
                    model_predicted_stage=model_predicted_stage,
                    ground_truth_stage=true_label,
                    base_rules="\n".join(self.base_rules)
                )
                messages.append({"role": "user", "content": creation_prompt})

                memory_output = self.get_schema_followed_response(
                    messages, schema_set["memory_creation"]
                )
                if memory_output: 
                    memory_text = memory_output["memory"].strip()
                    if memory_text:
                        self.all_memory.append(memory_text)
                    # Check if we need to refine memory
                    needs_refinement = (
                        len(self.all_memory) >= memory_threshold
                        or any(len(item) >= max_memory_item_length for item in self.all_memory)
                    )
                    if needs_refinement:
                        print(f"Memory refinement triggered at row {idx}.")
                        self.refine_memory(dataset, idx)

            pbar.update(1)
        pbar.close()

        print(f"\nTotal parsing errors: {parsing_error}")
        print(f"Total memory creation events: {len(memory_creation_indices)}")

        # For convenience, store the rules and final memory.
        dataset["base_rules"] = [self.base_rules] * len(dataset)
        dataset["final_memory"] = [self.all_memory] * len(dataset)

        # If you want to measure intervals:
        # memory_creation_indices might look like [5, 12, 25, 40, ...]
        # intervals are differences between consecutive indices
        memory_intervals = [
            j - i for i, j in zip(memory_creation_indices[:-1], memory_creation_indices[1:])
        ]
        print("Memory creation intervals:", memory_intervals)

        return dataset

    def refine_memory(self, dataset: pd.DataFrame, idx: int):
        """
        Calls the memory refinement prompt to shorten or reorganize the memory 
        if it becomes too large or unwieldy.
        """
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_memory_refinement.format(
                base_rules="\n".join(self.base_rules), all_accumulated_memory="\n".join(self.all_memory)
            )}
        ]
        refine_output = self.get_schema_followed_response(messages, schema_set["memory_refinement"])
        if refine_output and refine_output["refined_memory"]:
            refined = refine_output["refined_memory"]
            # Overwrite the all_memory list in-place
            self.all_memory.clear()
            self.all_memory.extend(refined)
            # Store this info in your dataset if you wish
            dataset.loc[idx, f"{self.test_name}_refine_memory_count"] = len(refined)
        else:
            print(f"Refinement failed at row {idx}, keeping old memory items.")



if __name__ == "__main__":
    # Initialize your OpenAI client
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1", timeout=120.0)
    
    # Load the dataset
    brca_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv")
    # We'll use the T-stage columns here. 
    df_base = brca_report[["patient_filename", "t", "text"]]

    # Instantiate your agent once
    agent = Agent(
        client=client,
        model="meta-llama/Llama-3.3-70B-Instruct",
        label="t",
        test_name="ltm_test"
    )

    # Run the experiment multiple times with different shuffles
    num_runs = 1  # or however many you need
    for run_idx in range(num_runs):
        # Shuffle with a new random seed each time
        # (Or you could use run_idx as the seed if you want reproducibility)
        random_seed = random.randint(0, 999999)
        print(f"Starting run {run_idx} with seed {random_seed}")  
        df_shuffled = df_base.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Run the combined experiment
        result_df = agent.run_experiment(df_shuffled)

        # Save to CSV with a run index in the file name
        output_filename = f"results_t_stage_run_{run_idx}.csv"
        result_df.to_csv(output_filename, index=False)

        print(f"Finished run {run_idx} - results saved to {output_filename}")