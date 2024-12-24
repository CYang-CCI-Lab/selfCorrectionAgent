import os
import json
import pandas as pd
from typing import List, Dict, Optional, Literal
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI

# ------------------------------------------------------
# Prompts
# ------------------------------------------------------
system_instruction = "You are an expert in determining the T stage of breast cancer following the AJCC TNM classification system."

prompt_generate_rules = """
Please articulate the official AJCC T staging rules for breast cancer, focusing only on T1, T2, T3, and T4 (ignore any sub-stages such as T1a, T1b, etc.).
Return your answer as a list of short statements.
"""

prompt_test_with_rules_and_memory = """
You are provided with:
1) Core T-staging rules the model generated from its own knowledge:
{rules}

2) Additional “memory” items accumulated from past mistakes or clarifications:
{memory}

Here is the pathology report:
{report}

Please:
1. Explain your reasoning step by step, explicitly referencing any relevant numeric cutoffs or special T4 criteria.
2. Provide your predicted T stage, choosing only from T1, T2, T3, or T4.
"""

prompt_memory_creation = """
Your previous reasoning and predicted T stage are:
- Reasoning: {model_reasoning}
- Your Predicted T stage: {model_predicted_stage}

The actual ground truth T stage for this report is: {ground_truth_T_stage}.

Please identify the specific step(s) where your reasoning diverged from the correct reasoning according to the official AJCC guidelines.
"""
# Please:
# 1. Pinpoint the specific step(s) where your reasoning diverged from the correct reasoning according to the official AJCC guidelines. 
# 2. Create or adjust your “memory” for more accurate T-stage determinations in similar cases, emphasizing numeric cutoffs and definitive criteria.
# 3. Ensure that your new memory does not include unverified or non-AJCC-based adjustments. Keep it concise, general, and aligned with the AJCC classification.

prompt_memory_refinement = """
Below is a large set of memory items accumulated from analyzing various pathology reports. 
They may include redundancies, overly specific or confusing guidance, or contradictory points.

Your task is to review and consolidate them into a smaller, cleaner, and more authoritative “memory” for determining T stages in breast cancer according to AJCC TNM by:

1. Removing any non-official or contradictory suggestions.
2. Clearly stating the numeric cutoffs.
3. Avoiding complicated margin-based adjustments or combined lesion size unless the AJCC guidelines explicitly support it.
4. Keeping your final memory organized, easy to follow, and free of duplicates.

Here is the accumulated memory:

{all_accumulated_memory}
"""

# ------------------------------------------------------
# Pydantic models (same as your examples)
# ------------------------------------------------------
class Response_T(BaseModel):
    reasoning: str = Field(
        description="A concise, step-by-step explanation of how the model arrived at the T stage."
    )
    stage: Literal["T1", "T2", "T3", "T4"] = Field(
        description="The T stage determined from the report."
    )

class Response_MemoryCreation_T(BaseModel):
    error_analysis: str = Field(
        description="Pinpoint the specific step(s) where reasoning deviated from official AJCC guidelines."
    )
    memory: List[str] = Field(
        description=(
            "A list of new or updated memory items. Must highlight the official AJCC rules. " 
            "Avoid unverified or contradictory suggestions."
        )
    )

class Response_MemoryRefinement_T(BaseModel):
    refined_memory: List[str] = Field(
        description=(
            "A smaller, cleaner set of memory items that fully conform to AJCC guidelines, "
            "clearly stating numeric cutoffs for T1–T4, etc."
        )
    )

schema_set = {
    "cold_start": Response_T.model_json_schema(),
    "memory_creation": Response_MemoryCreation_T.model_json_schema(),
    "memory_refinement": Response_MemoryRefinement_T.model_json_schema()
}

# ------------------------------------------------------
# Agent class
# ------------------------------------------------------
class Agent:
    def __init__(self, client: OpenAI, model: str, label: str, test_name: str) -> None:
        self.client = client
        self.model = model
        self.label = label
        self.test_name = test_name

        # Where we store the self-generated AJCC T-staging rules
        self.rules: str = ""  

    def generate_rules(self) -> str:
        """
        Asks the LLM to produce the official AJCC T-staging rules from its own knowledge,
        in a short bullet-point or paragraph form. We store it in self.rules.
        """
        # You can treat this as a standard system + user prompt
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_generate_rules}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # We only need plain text; no special schema for "rules."
                temperature=0.1
            )
            rules_text = response.choices[0].message.content.strip()
            self.rules = rules_text
            return rules_text
        except Exception as e:
            print(f"Error generating rules: {e}")
            return ""

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

    def refine_memory(self, all_memory: List[str], dataset: pd.DataFrame, idx: int):
        """
        Calls the memory refinement prompt to shorten or reorganize the memory 
        if it becomes too large or unwieldy.
        """
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": prompt_memory_refinement.format(all_accumulated_memory=all_memory)
            }
        ]

        refine_output = self.get_schema_followed_response(messages, schema_set["memory_refinement"])
        if refine_output:
            refined = refine_output["refined_memory"]
            # Overwrite the all_memory list in-place
            all_memory.clear()
            all_memory.extend(refined)
            # Optionally store this info in your dataset if you wish
            dataset.loc[idx, f"{self.test_name}_refine_memory_count"] = len(refined)
        else:
            print(f"Refinement failed at row {idx}, keeping old memory items.")

    def run_experiment_with_inherent_rules(
        self,
        dataset: pd.DataFrame,
        memory_threshold: int = 10,
        max_memory_item_length: int = 300
    ) -> pd.DataFrame:
        """
        1) Generate T-staging rules from the model’s own knowledge (store in self.rules).
        2) For each row:
           - Provide the LLM with both the self.rules (the inherent official T-stage guidelines) 
             and the accumulated memory from previous mistakes.
           - If the model's prediction is wrong, do memory creation.
           - Possibly refine memory if threshold is met.
        3) Return updated dataset with predictions, reasoning, etc.
        """
        # 1) Generate the AJCC T staging rules from LLM’s inherent knowledge
        if not self.rules:
            print("Generating T-staging rules from the LLM's inherent knowledge...")
            self.generate_rules()
            print("Rules generated:\n", self.rules)

        all_memory: List[str] = []
        parsing_error = 0
        memory_creation_indices = []

        pbar = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            report = row["text"]
            true_label = f"T{row[self.label]+1}"

            # 2) Use the self.rules + the memory
            memory_context = "\n".join(all_memory) if all_memory else "None so far."
            filled_prompt = prompt_test_with_rules_and_memory.format(
                rules=self.rules,
                memory=memory_context,
                report=report
            )

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": filled_prompt}
            ]

            # Reuse the same T-stage schema
            json_output = self.get_schema_followed_response(messages, schema_set["cold_start"])
            if not json_output:
                parsing_error += 1
                print(f"Error parsing JSON at row {idx}")
                dataset.loc[idx, f"{self.test_name}_parsed"] = False
                pbar.update(1)
                continue

            model_reasoning = json_output["reasoning"]
            model_predicted_stage = json_output["stage"]
            dataset.loc[idx, f"{self.test_name}_parsed"] = True
            dataset.loc[idx, f"{self.test_name}_reasoning"] = model_reasoning
            dataset.loc[idx, f"{self.test_name}_pred"] = model_predicted_stage

            # Check correctness
            if model_predicted_stage != true_label:
                # Trigger memory creation
                memory_creation_indices.append(idx)

                # Provide the model’s output as part of the conversation
                messages.append({"role": "assistant", "content": str(json_output)})

                creation_prompt = prompt_memory_creation.format(
                    model_reasoning=model_reasoning,
                    model_predicted_stage=model_predicted_stage,
                    ground_truth_T_stage=true_label
                )
                messages.append({"role": "user", "content": creation_prompt})

                creation_output = self.get_schema_followed_response(
                    messages, schema_set["memory_creation"]
                )
                if creation_output:
                    # Extract new memory items
                    new_memory = creation_output["memory"]
                    dataset.loc[idx, f"{self.test_name}_error_analysis"] = creation_output["error_analysis"]
                    # Accumulate
                    all_memory.extend(new_memory)

                    # Check if we need to refine memory
                    needs_refinement = (
                        len(all_memory) >= memory_threshold
                        or any(len(item) >= max_memory_item_length for item in all_memory)
                    )
                    if needs_refinement:
                        print(f"Memory refinement triggered at row {idx}.")
                        self.refine_memory(all_memory, dataset, idx)

            pbar.update(1)
        pbar.close()

        print(f"\nTotal parsing errors: {parsing_error}")
        print(f"Total memory creation events: {len(memory_creation_indices)}")

        # Optionally store final memory in the dataset
        dataset["final_memory"] = [all_memory] * len(dataset)

        # Compute intervals
        memory_intervals = [
            j - i for i, j in zip(memory_creation_indices[:-1], memory_creation_indices[1:])
        ]
        print("Memory creation intervals:", memory_intervals)

        return dataset

    # (You can keep or adapt any existing baseline methods, etc.)
    def test_baseline(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Baseline testing method that:
        1) Uses only a cold start prompt for every row from start to end.
        2) Records the model's reasoning and T stage predictions.
        3) Never triggers any memory creation or refinement.
        """
        parsing_error = 0
        from_x = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            report = row["text"]
            # In baseline, we just use the original cold_start prompt (or any simple prompt)
            cold_prompt = f"Here is the pathology report:\n{report}\nProvide T stage (T1, T2, T3, T4)."
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": cold_prompt}
            ]
            json_output = self.get_schema_followed_response(messages, schema_set["cold_start"])
            if not json_output:
                parsing_error += 1
                dataset.loc[idx, f"{self.test_name}_parsed"] = False
                from_x.update(1)
                continue

            dataset.loc[idx, f"{self.test_name}_parsed"] = True
            dataset.loc[idx, f"{self.test_name}_reasoning"] = json_output["reasoning"]
            dataset.loc[idx, f"{self.test_name}_pred"] = json_output["stage"]
            from_x.update(1)
        from_x.close()

        print(f"Total parsing errors (baseline): {parsing_error}")
        return dataset


# ------------------------------------------------------
# Example usage
# ------------------------------------------------------
if __name__ == "__main__":
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1", timeout=120.0)

    brca_report = pd.read_csv(
        "/secure/shared_data/rag_tnm_results/summary/5_folds_summary/brca_df.csv"
    )
    df = brca_report[brca_report["n"] != -1][["patient_filename", "t", "text"]]

    # Instantiate agent
    agent = Agent(
        client=client,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        label="t",
        test_name="inherent_rules_test"
    )

    # 1) Use the new method that starts by generating T-staging rules from the LLM itself
    result_df = agent.run_experiment_with_inherent_rules(
        dataset=df,
        memory_threshold=5,
        max_memory_item_length=250
    )

    print(result_df)
    result_df.to_csv("1221_result_inherent_rules.csv", index=False)

    # 2) Optionally compare to your existing baseline
    baseline_agent = Agent(
        client=client,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        label="t",
        test_name="baseline_test"
    )
    baseline_results = baseline_agent.test_baseline(df)
    baseline_results.to_csv("baseline_results.csv", index=False)
