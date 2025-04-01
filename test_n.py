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

system_instruction = "You are an expert in determining the N stage of {cancer_type} cancer following the AJCC Cancer Staging Manual (7th edition)."

# Prompt that asks model to create rules from its own memory (LLM-based)
prompt_generate_llm_rules = """
Please articulate the N staging rules for {cancer_type} cancer, focusing only on N0, N1, N2, and N3.
Ignore any sub-stages.
Return your answer as a list of short statements that align strictly with the AJCC Cancer Staging Manual (7th edition).
"""

# Prompt that asks model to create rules from a RAG-based excerpt
prompt_generate_rag_rules = """
You are provided with an excerpt from the AJCC Cancer Staging Manual (7th edition) for {cancer_type} cancer, focusing on N stage classification.

Excerpt:
{rag_context}

Please:
1. Summarize the guidelines for N0, N1, N2, and N3 (no sub-stages) in bullet form.
2. Return your final summary as a list of short statements that align strictly with the excerpt.
"""

# Zero-shot test
prompt_test_zero_shot = """
You are provided with a pathology report of a {cancer_type} cancer patient.

Pathology report:
{report}

Please:
1. Explain your reasoning step by step as you determine the N stage.
2. Provide your final predicted N stage (N0, N1, N2, or N3).
"""

# Memory-only test
prompt_test_memory_only = """
You are provided with a set of memory items (lessons learned from past mistakes or borderline cases) and a pathology report for a {cancer_type} cancer patient.

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
You are provided with official N staging rules for {cancer_type} cancer (N0, N1, N2, N3), derived from the AJCC Cancer Staging Manual (7th edition), 
and a pathology report for a {cancer_type} cancer patient.

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
You are provided with official N staging rules for {cancer_type} cancer (N0, N1, N2, N3), derived from the AJCC Cancer Staging Manual (7th edition), 
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
You are provided with relevant context retrieved from the AJCC Cancer Staging Manual (7th edition) for {cancer_type} cancer (N0, N1, N2, N3),
and a pathology report for a {cancer_type} cancer patient.

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
# Pydantic models
# ----------------------------------------------------------------------

class ResponseRules(BaseModel):
    rules: List[str] = Field(
        description="N staging rules (N0, N1, N2, N3) as a list of short statements."
    )

class ResponseStage(BaseModel):
    reasoning: str = Field(
        description="A step-by-step explanation for how the model arrived at the predicted N stage."
    )
    stage: Literal["N0", "N1", "N2", "N3"] = Field(
        description="The final predicted N stage (N0, N1, N2, or N3)."
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
        description="A cleaner set of memory items."
    )

schema_set = {
    "rule_generation": ResponseRules.model_json_schema(),
    "stage_prediction": ResponseStage.model_json_schema(),
    "memory_creation": ResponseMemoryCreation.model_json_schema(),
    "memory_refinement": ResponseMemoryRefinement.model_json_schema()
}


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------
class Agent:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        cancer_type: str,
        label: str,
        test_name: str,
        context_file: str = "context.json"
    ):
        self.client = client
        self.model = model
        self.cancer_type = cancer_type

        if "t" in label.lower():
            self.label = "t"
        elif "n" in label.lower():
            self.label = "n"
        self.test_name = test_name
        self.context_file = context_file

        # We'll store two sets of rules:
        self.llm_rules: List[str] = []  # LLM-based
        self.rag_rules: List[str] = []  # RAG-based

        self.all_memory_from_llm_rules: List[str] = []
        self.all_memory_from_rag_rules: List[str] = []

        # We'll load the raw RAG excerpt from file
        self.rag_context = self.load_rag_context()

    def load_rag_context(self) -> str:
        with open(self.context_file, "r") as f:
            data = json.load(f)
        if self.label == "t":
            return data["rag_raw_t14"]
        elif self.label == "n":
            return data["rag_raw_n03"]

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
            print(f"Error: {e}")
            return None

    def run_experiment(
        self,
        dataset: pd.DataFrame,
        memory_threshold: int = 1,
        max_memory_item_length: int = 1
    ) -> pd.DataFrame:
        """
        We run 7 total methods:
          1) Zero-shot
          2) LLM-based rules-only
          3) RAG-based rules-only
          4) Memory-only
          5) RAG-only (excerpt-only)
          6) LLM-based rules + memory
          7) RAG-based rules + memory
        """

        # ---------------------------------------------------------
        # 1) Generate LLM-based rules (via prompt_generate_llm_rules)
        # ---------------------------------------------------------
        print("Generating LLM-based rules...")
        messages_llm_rules = [
            {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
            {"role": "user", "content": prompt_generate_llm_rules.format(cancer_type=self.cancer_type)}
        ]
        llm_rules_response = self.get_schema_followed_response(messages_llm_rules, schema_set["rule_generation"])
        if llm_rules_response:
            self.llm_rules = llm_rules_response["rules"]  # from ResponseRules
        else:
            print("Error: Could not generate LLM-based rules.")
            return dataset

        print("LLM-based rules:", self.llm_rules)

        # ---------------------------------------------------------
        # 2) Generate RAG-based rules (via prompt_generate_rag_rules)
        # ---------------------------------------------------------
        print("Generating RAG-based rules...")
        rag_prompt_filled = prompt_generate_rag_rules.format(rag_context=self.rag_context, cancer_type=self.cancer_type)
        messages_rag_rules = [
            {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
            {"role": "user", "content": rag_prompt_filled}
        ]
        rag_rules_response = self.get_schema_followed_response(messages_rag_rules, schema_set["rule_generation"])
        if rag_rules_response:
            self.rag_rules = rag_rules_response["rules"]  # from ResponseRules
        else:
            print("Error: Could not generate RAG-based rules.")
            return dataset

        print("RAG-based rules:", self.rag_rules)

        # Prepare columns
        dataset["current_memory_from_llm"] = None
        dataset["current_memory_from_llm"] = dataset["current_memory_from_llm"].astype(object)
        dataset["current_memory_from_rag"] = None
        dataset["current_memory_from_rag"] = dataset["current_memory_from_rag"].astype(object)

        parsing_error = 0

        pbar = tqdm(total=dataset.shape[0])
        for idx, row in dataset.iterrows():
            report = row["text"]
            if self.label == "t":
                true_label = f"T{row[self.label]+1}"  # e.g. T1, T2, T3, T4
            elif self.label == "n":
                true_label = f"N{row[self.label]}"  # e.g. N0, N1, N2, N3

            # Store memory at this row
            dataset.at[idx, "current_memory_from_llm"] = copy.deepcopy(self.all_memory_from_llm_rules)
            dataset.at[idx, "current_memory_from_rag"] = copy.deepcopy(self.all_memory_from_rag_rules)

            # -------------------------------------------------
            # Baseline #1: Zero-shot
            # -------------------------------------------------
            zs_prompt = prompt_test_zero_shot.format(report=report, cancer_type=self.cancer_type)
            msgs_zs = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": zs_prompt}
            ]
            zs_resp = self.get_schema_followed_response(msgs_zs, schema_set["stage_prediction"])
            if zs_resp:
                dataset.loc[idx, f"{self.test_name}_pred_zs"] = zs_resp["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_zs"] = zs_resp["reasoning"]

            # -------------------------------------------------
            # Baseline #2-1: LLM-based rules-only
            # -------------------------------------------------
            llm_rules_str = "\n".join(self.llm_rules)
            llm_rules_prompt = prompt_test_with_base_rules.format(
                report=report,
                base_rules=llm_rules_str,
                cancer_type=self.cancer_type
            )
            msgs_llm_rulesonly = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": llm_rules_prompt}
            ]
            llm_rulesonly_resp = self.get_schema_followed_response(msgs_llm_rulesonly, schema_set["stage_prediction"])
            if llm_rulesonly_resp:
                dataset.loc[idx, f"{self.test_name}_pred_llm_rulesonly"] = llm_rulesonly_resp["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_llm_rulesonly"] = llm_rulesonly_resp["reasoning"]

            # -------------------------------------------------
            # Baseline #2-2: RAG-based rules-only
            # -------------------------------------------------
            rag_rules_str = "\n".join(self.rag_rules)
            rag_rules_prompt = prompt_test_with_base_rules.format(
                report=report,
                base_rules=rag_rules_str,
                cancer_type=self.cancer_type
            )
            msgs_rag_rulesonly = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": rag_rules_prompt}
            ]
            rag_rulesonly_resp = self.get_schema_followed_response(msgs_rag_rulesonly, schema_set["stage_prediction"])
            if rag_rulesonly_resp:
                dataset.loc[idx, f"{self.test_name}_pred_rag_rulesonly"] = rag_rulesonly_resp["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_rag_rulesonly"] = rag_rulesonly_resp["reasoning"]

            # -------------------------------------------------
            # Baseline #3-1: Memory-only (from LLM-based rules)
            # -------------------------------------------------
            memory_text = "\n".join(self.all_memory_from_llm_rules) if self.all_memory_from_llm_rules else "No prior memory."
            memonly_prompt = prompt_test_memory_only.format(
                report=report,
                all_memory=memory_text,
                cancer_type=self.cancer_type
            )
            msgs_memonly = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": memonly_prompt}
            ]
            memonly_resp = self.get_schema_followed_response(msgs_memonly, schema_set["stage_prediction"])
            if memonly_resp:
                dataset.loc[idx, f"{self.test_name}_pred_memonly"] = memonly_resp["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_memonly"] = memonly_resp["reasoning"]

            # -------------------------------------------------
            # Baseline #3-2: Memory-only (from RAG-based rules)
            # -------------------------------------------------
            memory_text2 = "\n".join(self.all_memory_from_rag_rules) if self.all_memory_from_rag_rules else "No prior memory."
            memonly_prompt2 = prompt_test_memory_only.format(
                report=report,
                all_memory=memory_text2,
                cancer_type=self.cancer_type
            )
            msgs_memonly2 = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": memonly_prompt2}
            ]
            memonly_resp2 = self.get_schema_followed_response(msgs_memonly2, schema_set["stage_prediction"])
            if memonly_resp2:
                dataset.loc[idx, f"{self.test_name}_pred_memonly2"] = memonly_resp2["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_memonly2"] = memonly_resp2["reasoning"]

            # -------------------------------------------------
            # Baseline #4: RAG-only (excerpt-only)
            # -------------------------------------------------
            rag_only_prompt = prompt_test_rag.format(
                rag_context=self.rag_context,
                report=report,
                cancer_type=self.cancer_type
            )
            msgs_rag_only = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": rag_only_prompt}
            ]
            rag_only_resp = self.get_schema_followed_response(msgs_rag_only, schema_set["stage_prediction"])
            if rag_only_resp:
                dataset.loc[idx, f"{self.test_name}_pred_rag_only"] = rag_only_resp["stage"]
                dataset.loc[idx, f"{self.test_name}_reasoning_rag_only"] = rag_only_resp["reasoning"]

            # -------------------------------------------------
            # Main #1: LLM-based rules + memory
            # -------------------------------------------------
            if not self.all_memory_from_llm_rules:
                llm_mem_prompt = prompt_test_with_base_rules.format(
                    report=report,
                    base_rules=llm_rules_str,
                    cancer_type=self.cancer_type
                )
            else:
                llm_mem_prompt = prompt_test_with_rules_and_memory.format(
                    report=report,
                    base_rules=llm_rules_str,
                    all_memory="\n".join(self.all_memory_from_llm_rules),
                    cancer_type=self.cancer_type
                )
            msgs_llm_mem = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": llm_mem_prompt}
            ]
            llm_mem_resp = self.get_schema_followed_response(msgs_llm_mem, schema_set["stage_prediction"])
            if not llm_mem_resp:
                parsing_error += 1
                print(f"Error at row {idx} (LLM-based rules + memory).")
                pbar.update(1)
                continue

            dataset.loc[idx, f"{self.test_name}_pred_llm_mem"] = llm_mem_resp["stage"]
            dataset.loc[idx, f"{self.test_name}_reasoning_llm_mem"] = llm_mem_resp["reasoning"]

            # Memory creation if mismatch
            if llm_mem_resp["stage"] != true_label:
                msgs_llm_mem.append({"role": "assistant", "content": str(llm_mem_resp)})

                creation_prompt = prompt_memory_creation.format(
                    model_reasoning=llm_mem_resp["reasoning"],
                    model_predicted_stage=llm_mem_resp["stage"],
                    ground_truth_stage=true_label,
                    base_rules=llm_rules_str
                )
                msgs_llm_mem.append({"role": "user", "content": creation_prompt})
                mem_creation_resp = self.get_schema_followed_response(msgs_llm_mem, schema_set["memory_creation"])
                if mem_creation_resp:
                    new_mem = mem_creation_resp["memory"].strip()
                    if new_mem:
                        self.all_memory_from_llm_rules.append(new_mem)

                    # Possibly refine
                    needs_refinement = (
                        len(self.all_memory_from_llm_rules) >= memory_threshold
                        or any(len(item) >= max_memory_item_length for item in self.all_memory_from_llm_rules)
                    )
                    if needs_refinement:
                        print(f"Memory refinement triggered at row {idx} (LLM-based rules + memory).")
                        print(f"Before refinement, length of memory: {len(self.all_memory_from_llm_rules)}")
                        self.refine_memory(dataset, idx, llm_rules_str, self.all_memory_from_llm_rules)
                        print(f"After refinement, length of memory: {len(self.all_memory_from_llm_rules)}")

            # -------------------------------------------------
            # Main #2: RAG-based rules + memory
            # -------------------------------------------------
            if not self.all_memory_from_rag_rules:
                rag_mem_prompt = prompt_test_with_base_rules.format(
                    report=report,
                    base_rules=rag_rules_str,
                    cancer_type=self.cancer_type
                )
            else:
                rag_mem_prompt = prompt_test_with_rules_and_memory.format(
                    report=report,
                    base_rules=rag_rules_str,
                    all_memory="\n".join(self.all_memory_from_rag_rules),
                    cancer_type=self.cancer_type
                )
            msgs_rag_mem = [
                {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
                {"role": "user", "content": rag_mem_prompt}
            ]
            rag_mem_resp = self.get_schema_followed_response(msgs_rag_mem, schema_set["stage_prediction"])
            if not rag_mem_resp:
                parsing_error += 1
                print(f"Error at row {idx} (RAG-based rules + memory).")
                pbar.update(1)
                continue

            dataset.loc[idx, f"{self.test_name}_pred_rag_mem"] = rag_mem_resp["stage"]
            dataset.loc[idx, f"{self.test_name}_reasoning_rag_mem"] = rag_mem_resp["reasoning"]

            # Memory creation if mismatch
            if rag_mem_resp["stage"] != true_label:
                msgs_rag_mem.append({"role": "assistant", "content": str(rag_mem_resp)})

                creation_prompt2 = prompt_memory_creation.format(
                    model_reasoning=rag_mem_resp["reasoning"],
                    model_predicted_stage=rag_mem_resp["stage"],
                    ground_truth_stage=true_label,
                    base_rules=rag_rules_str
                )
                msgs_rag_mem.append({"role": "user", "content": creation_prompt2})
                mem_creation_resp2 = self.get_schema_followed_response(msgs_rag_mem, schema_set["memory_creation"])
                if mem_creation_resp2:
                    new_mem2 = mem_creation_resp2["memory"].strip()
                    if new_mem2:
                        self.all_memory_from_rag_rules.append(new_mem2)

                    # Possibly refine
                    needs_refinement = (
                        len(self.all_memory_from_rag_rules) >= memory_threshold
                        or any(len(item) >= max_memory_item_length for item in self.all_memory_from_rag_rules)
                    )
                    if needs_refinement:
                        print(f"Memory refinement triggered at row {idx} (RAG-based rules + memory).")
                        print(f"Before refinement, length of memory: {len(self.all_memory_from_rag_rules)}")
                        self.refine_memory(dataset, idx, rag_rules_str, self.all_memory_from_rag_rules)
                        print(f"After refinement, length of memory: {len(self.all_memory_from_rag_rules)}")

            pbar.update(1)

        pbar.close()
        print(f"Total parsing errors: {parsing_error}")

        # Save final memory + rules
        dataset["llm_rules"] = [self.llm_rules] * len(dataset)
        dataset["rag_rules"] = [self.rag_rules] * len(dataset)
        dataset["final_memory_from_llm"] = [self.all_memory_from_llm_rules] * len(dataset)
        dataset["final_memory_from_rag"] = [self.all_memory_from_rag_rules] * len(dataset)

        return dataset

    def refine_memory(self, dataset: pd.DataFrame, idx: int, base_rules_str: str, all_memory: List[str]):
        """
        Memory refinement: pass the same rules used for the approach that triggered refinement.
        """
        msgs_ref = [
            {"role": "system", "content": system_instruction.format(cancer_type=self.cancer_type)},
            {
                "role": "user",
                "content": prompt_memory_refinement.format(
                    all_accumulated_memory="\n".join(all_memory),
                    base_rules=base_rules_str
                )
            }
        ]
        ref_resp = self.get_schema_followed_response(msgs_ref, schema_set["memory_refinement"])
        if ref_resp and ref_resp["refined_memory"]:
            all_memory.clear()
            all_memory.extend(ref_resp["refined_memory"])
        else:
            print(f"Refinement failed at row {idx}, keeping old memory items.")


# ------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------
if __name__ == "__main__":
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1", timeout=120.0)

    lung_report = pd.read_csv("/secure/shared_data/rag_tnm_results/summary/5_folds_summary/luad_df.csv")
    df_base = lung_report[lung_report["n"] != -1][["patient_filename", "n", "text"]]

    agent = Agent(
        client=client,
        model="meta-llama/Llama-3.3-70B-Instruct",
        cancer_type="lung",
        label="n",
        test_name="experiment_n_stage",
        context_file="context_lung.json",
    )

    num_runs = 5
    for run_idx in range(num_runs):
        random_seed = random.randint(0, 999999)
        print(f"Starting run {run_idx} with random seed {random_seed}...")
        df_shuffled = df_base.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        result_df = agent.run_experiment(df_shuffled)

        output_filename = f"results_n_stage_run_{run_idx}_lung.csv"
        result_df.to_csv(output_filename, index=False)
        print(f"Finished run {run_idx}. Results saved to {output_filename}")
