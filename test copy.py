import os
import json
import pandas as pd
from typing import List, Dict, Optional, Literal
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI

system_instruction = "You are an expert in determining the T stage of breast cancer following the AJCC TNM classification system."

prompt_generate_rules = """
Please articulate the official AJCC T staging rules for breast cancer, focusing only on T1, T2, T3, and T4 (ignore any sub-stages such as T1a, T1b, etc.).

Return your answer as a list of short statements.
"""

prompt_test_with_base_rules = """
You are provided with a pathology report of a breast cancer patient and the T staging rules for breast cancer.
Your task is to determine the T stage of the patient based on the report and the rules.



Please:
1. Explain your reasoning step by step as you determine the T stage.
2. Provide your predicted T stage, choosing from T1, T2, T3, T4.
"""

prompt_memory_creation = """
Your previous reasoning and predicted T stage are:
- Reasoning: {model_reasoning}
- Predicted T stage: {model_predicted_stage}

The actual ground truth T stage is: {ground_truth_T_stage}.

Here are the guidelines for determining the T stage:

{base_rules}

Where did your reasoning diverge from these rules (or from the correct ground truth),
and how can you refine your approach? Please create new memory items 
that specifically address this error, without contradicting your own base rules.

Please:
1. Pinpoint the specific step(s) where your reasoning and prediction diverged from the correct T stage according to the guidelines. 
2. Create your “memory” for more accurate T-stage determinations in similar cases, emphasizing numeric cutoffs and definitive criteria.
3. Ensure that your memory does not include unverified or non-AJCC-based adjustments. Keep it concise, general, and aligned with the AJCC classification.
"""

prompt_test_with_rules_and_memory = """
Here are the guidelines for determining the T stage of breast cancer:
{base_rules}

Here is your accumulated memory of past mistakes and lessons:
{all_memory}

Now, here is the pathology report:
{report}

Please:
1. Refer to the guidelines and your memory as you determine the T stage.
2. Explain your reasoning step by step.
3. Provide your predicted T stage, choosing from T1, T2, T3, T4.
"""

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

###################

prompt_cold_start = """
Here is the pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the T stage 
   (according to AJCC TNM guidelines).
2. Provide your predicted T stage, choosing from T1, T2, T3, or T4.
"""

prompt_memory_creation = """
Your previous reasoning and predicted T stage are:
- Reasoning: {model_reasoning}
- Your Predicted T stage: {model_predicted_stage}

The actual ground truth T stage for this report is: {ground_truth_T_stage}.

Please:
1. Pinpoint the specific step(s) where your reasoning diverged from the correct reasoning according to the official AJCC guidelines. 
2. Create or adjust your “memory” for more accurate T-stage determinations in similar cases, emphasizing numeric cutoffs and definitive criteria.
3. Ensure that your new memory does not include unverified or non-AJCC-based adjustments. Keep it concise, general, and aligned with the AJCC classification.
"""

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

prompt_test_with_memory = """
You are provided with a pathology report for a breast cancer patient and a set of memory items that guide T stage determination.

Here is the relevant memory:
{context}

Here is the pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the T stage, ensuring you follow these memory guidelines.
2. Provide your predicted T stage, choosing from T1, T2, T3, T4.
"""

# ------------------------------------------------------
# Pydantic models (minimal changes from your examples)
# ------------------------------------------------------

class Response_T(BaseModel):
    reasoning: str = Field(
        description=(
            "A concise, step-by-step explanation of how the model arrived at the predicted "
            "T stage, explicitly referencing numeric cutoffs or crieteria."
        )
    )
    stage: Literal["T1", "T2", "T3", "T4"] = Field(
        description=(
            "The T stage determined from the report. Must be one of 'T1', 'T2', 'T3', or 'T4'. "
            "No additional sub-stages or variations allowed."
        )
    )

class Response_MemoryCreation_T(BaseModel):
    error_analysis: str = Field(
        description=(
            "Pinpoint the specific step(s) where the model's reasoning deviated from official AJCC guidelines."
        )
    )
    memory: List[str] = Field(
        description=(
            "A list of new or updated memory items (concise guidelines). Must highlight the official AJCC rules. " 
            "Avoid unverified or contradictory suggestions. "
            "No more than a few points, each referencing the AJCC classification."
        )
    )

class Response_MemoryRefinement_T(BaseModel):
    refined_memory: List[str] = Field(
        description=(
            "A smaller, cleaner set of memory items that fully conform to AJCC guidelines. "
            "It should clearly state the numeric cutoffs for T1, T2, T3, and T4. "
            "Avoid extraneous case-specific details, conflicting notes, or any suggestion to arbitrarily adjust T stage."
        )
    )


schema_set = {
    "cold_start": Response_T.model_json_schema(),
    "memory_creation": Response_MemoryCreation_T.model_json_schema(),
    "memory_refinement": Response_MemoryRefinement_T.model_json_schema(),
}

# ------------------------------------------------------
# Example Agent
# ------------------------------------------------------
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
        memory_threshold: int = 10,   # e.g., refine if we have >=10 memory items
        max_memory_item_length: int = 300  # e.g., refine if any item is over 300 chars
    ) -> pd.DataFrame:
        """
        Interleaves testing and memory creation in a single pass.
        1) Start with empty memory.
        2) For each row:
            - If no memory, use cold start prompt.
            - Else use prompt_test_with_memory.
            - If the model's prediction is wrong, do memory creation.
            - Accumulate new memory and possibly refine if threshold is exceeded.
        3) Keep track of intervals between memory creation triggers to see 
           if they grow as memory improves staging accuracy.

        Returns the updated dataset with predictions, reasoning, etc.
        """
        all_memory: List[str] = []
        parsing_error = 0
        memory_creation_indices = []  # to store row indices where memory creation happened

        pbar = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            report = row["text"]
            true_label = f"T{row[self.label]+1}"

            # If no memory, use cold start prompt
            if not all_memory:
                filled_prompt = prompt_cold_start.format(report=report)
                schema_for_answer = schema_set["cold_start"]
            else:
                # If we have memory, use test with memory
                memory_context = "\n".join(all_memory)
                filled_prompt = prompt_test_with_memory.format(
                    context=memory_context,
                    report=report
                )
                schema_for_answer = schema_set["cold_start"]  # same structure as cold_start

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": filled_prompt}
            ]

            # Get response
            json_output = self.get_schema_followed_response(messages, schema_for_answer)
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

        # For convenience, store the final memory somewhere (optional).
        dataset["final_memory"] = [all_memory] * len(dataset)

        # If you want to measure intervals:
        # memory_creation_indices might look like [5, 12, 25, 40, ...]
        # intervals are differences between consecutive indices
        memory_intervals = [
            j - i for i, j in zip(memory_creation_indices[:-1], memory_creation_indices[1:])
        ]
        print("Memory creation intervals:", memory_intervals)

        return dataset

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

    def test_baseline(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Baseline testing method that:
        1) Uses only the cold start prompt for every row from start to end.
        2) Records the model's reasoning and T stage predictions.
        3) Never triggers any memory creation or refinement.
        
        Returns the updated DataFrame containing parsing status, reasoning, and predictions.
        """
        parsing_error = 0

        pbar = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            report = row["text"]
            # Build the cold start prompt with the pathology report
            filled_prompt = prompt_cold_start.format(report=report)
            
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": filled_prompt}
            ]
            
            # Get JSON output from the LLM (using the appropriate schema for T stage)
            json_output = self.get_schema_followed_response(messages, schema_set["cold_start"])
            if not json_output:
                parsing_error += 1
                print(f"Error parsing JSON at row {idx}")
                dataset.loc[idx, f"{self.test_name}_parsed"] = False
                continue
            
            # Record the model's reasoning and predicted stage
            dataset.loc[idx, f"{self.test_name}_parsed"] = True
            dataset.loc[idx, f"{self.test_name}_reasoning"] = json_output["reasoning"]
            dataset.loc[idx, f"{self.test_name}_pred"] = json_output["stage"]
            pbar.update(1)
        pbar.close()

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


    # # Instantiate agent
    # agent = Agent(
    #     client=client,
    #     model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    #     label="t",
    #     test_name="memory_test"
    # )

    # # Run the combined experiment
    # result_df = agent.run_experiment(
    #     dataset=df,
    #     memory_threshold=5,           # refine after 5 memory items 
    #     max_memory_item_length=250    # or if any memory item is >250 chars
    # )

    # # Inspect results
    # print(result_df)
    # result_df.to_csv("1221_result.csv", index=False)
    # # The intervals between memory creation events will be printed. 
    # # If they become larger (e.g., from 5 to 12 to 30, etc.), 
    # # that indicates memory is improving the model's accuracy over time.

    agent = Agent(
        client=client,
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        label="t",
        test_name="t14_baseline"
    )

    baseline_results = agent.test_baseline(df)
    baseline_results.to_csv("baseline_results.csv", index=False)
