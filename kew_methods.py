from __future__ import annotations
import logging
import httpx
from utils import setup_logging, get_logger, safe_json_load
logger = get_logger()
from pathlib import Path
import argparse
import json
import os
import random
import re
import sys
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field



import pandas as pd
import numpy as np
from tqdm import tqdm

from fuzzywuzzy import fuzz  

from openai import OpenAI  


# ---------------------------
# Stage helpers
# ---------------------------

def idx_to_stage(task: str, idx: int) -> str:
    if task.lower() == "t":
        return f"T{idx + 1}"
    elif task.lower() == "n":
        return f"N{idx}"
    raise ValueError("task must be 't' or 'n'")


_T_RE = re.compile(r"T\s*([1-4])(?:\s*[A-D])?(?!\s*\d)", re.IGNORECASE)
_N_RE = re.compile(r"N\s*([0-3])(?:\s*[A-D])?(?!\s*\d)", re.IGNORECASE)

def stage_to_idx(task: str, stage) -> Optional[int]:
    """텍스트 어디에 있든 T/N 스테이지 토큰을 '부분 일치'로 찾아 정규화.
    T -> 0..3 (T1..T4), N -> 0..3 (N0..N3). 못 찾으면 None.
    """
    if pd.isna(stage):
        return None
    s = str(stage)
    if not s.strip():
        return None

    if task.lower() == "t":
        m = _T_RE.search(s)
        return (int(m.group(1)) - 1) if m else None
    elif task.lower() == "n":
        m = _N_RE.search(s)
        return int(m.group(1)) if m else None
    return None

def _coerce_list(cell) -> List[str]:
    """Parse a cell that may be a list or a stringified list into a Python list[str]."""
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if cell is None or pd.isna(cell):
        return []
    s = str(cell)
    # Try JSON then literal_eval
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    # Fallback: consider it a single rule/memory line if non-empty
    s = s.strip()
    return [s] if s else []

def _is_missing_stage(val, *, task: str, treat_unparseable: bool) -> bool:
    """True if the stage is empty/NaN; optionally also when parsing to idx fails."""
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if treat_unparseable and stage_to_idx(task, val) is None:
        return True
    return False

def _system_prompt_for(task: str, cancer_type: str) -> str:
    return SYSTEM_T.format(cancer_type=cancer_type) if task == "t" else SYSTEM_N.format(cancer_type=cancer_type)

def _labels_for(task: str) -> str:
    return "{T1, T2, T3, T4}" if task == "t" else "{N0, N1, N2, N3}"

def repair_missing_predictions(
    method: str,
    llm: LLMClient,
    cfg: RunConfig,
    df: pd.DataFrame,
    *,
    max_retries: int = 2,
    treat_unparseable_as_missing: bool = True,
    rag_context: Optional[str] = None,
) -> pd.DataFrame:
    """
    Re-run ONLY rows whose {method}_stage is missing, and update cells in-place.
    Returns a full DataFrame with fewer missing stage values.
    """
    out = df.copy()
    method = method.lower()
    assert method in {"zscot", "rag", "kewrag", "kewltm"}

    stage_col = f"{method}_stage"
    reason_col = f"{method}_reasoning"
    raw_col = f"{method}_raw_llm"

    if stage_col not in out.columns:
        raise ValueError(f"'{stage_col}' column not found in the input results CSV.")

    # mask of rows to repair
    mask = out[stage_col].apply(lambda v: _is_missing_stage(v, task=cfg.task, treat_unparseable=treat_unparseable_as_missing))
    # KEwLTM: never try to predict training rows
    if method == "kewltm" and "is_train" in out.columns:
        mask = mask & (~out["is_train"].fillna(False))

    target_idxs = out.index[mask].tolist()
    if not target_idxs:
        logger.info("[REPAIR] No rows to repair for method=%s.", method)
        return out

    sys_prompt = _system_prompt_for(cfg.task, cfg.cancer_type)
    labels = _labels_for(cfg.task)

    for i in tqdm(target_idxs, desc=f"REPAIR-{method.upper()}"):
        report = str(out.at[i, "text"])  # must exist in results CSV

        for attempt in range(1, max_retries + 1):
            if method == "zscot":
                prompt = PROMPT_ZSCOT.format(
                    cancer_type=cfg.cancer_type,
                    report=report,
                    task_upper=cfg.task.upper(),
                    labels=labels,
                )
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ]
                data, raw = llm.json_chat(messages, schema=stage_schema(cfg.task))
                reasoning = data.get("reasoning", "") if data else ""
                stage = data.get("stage", "") if data else ""

            elif method == "rag":
                if not rag_context:
                    raise ValueError("RAG repair requires --context-file (rag_context).")
                prompt = PROMPT_RAG_ONLY.format(
                    cancer_type=cfg.cancer_type,
                    rag_context=rag_context,
                    report=report,
                    task_upper=cfg.task.upper(),
                    labels=labels,
                )
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ]
                data, raw = llm.json_chat(messages, schema=stage_schema(cfg.task))
                reasoning = data.get("reasoning", "") if data else ""
                stage = data.get("stage", "") if data else ""

            elif method == "kewrag":
                rules = _coerce_list(out.at[i, "kewrag_rules"]) if "kewrag_rules" in out.columns else []
                # If rules are missing, we can still proceed with a bland placeholder,
                # or regenerate from rag_context if provided.
                if not rules and rag_context:
                    # regenerate lightweight rules once per repair call (optional)
                    prompt_gen = PROMPT_GENERATE_RULES_FROM_RAG.format(
                        cancer_type=cfg.cancer_type,
                        rag_context=rag_context,
                        task_upper=cfg.task.upper(),
                        labels=labels,
                    )
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt_gen},
                    ]
                    data_rules, _ = llm.json_chat(messages, schema=rules_schema())
                    rules = [r.strip() for r in (data_rules.get("rules") or []) if isinstance(r, str)] if data_rules else []
                rules_text = "\n".join(rules) if rules else "No usable rules."

                prompt = PROMPT_RULES_ONLY.format(
                    cancer_type=cfg.cancer_type,
                    rules=rules_text,
                    report=report,
                    task_upper=cfg.task.upper(),
                    labels=labels,
                )
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ]
                data, raw = llm.json_chat(messages, schema=stage_schema(cfg.task))
                reasoning = data.get("reasoning", "") if data else ""
                stage = data.get("stage", "") if data else ""

            else:  # method == "kewltm"
                memory = _coerce_list(out.at[i, "kewltm_memory"]) if "kewltm_memory" in out.columns else []
                prompt = PROMPT_INFER_WITH_MEMORY.format(
                    memory="\n".join(memory) if memory else "(empty)",
                    report=report,
                    task_upper=cfg.task.upper(),
                    labels=labels,
                )
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ]
                data, raw = llm.json_chat(messages, schema=stage_schema(cfg.task))
                reasoning = data.get("reasoning", "") if data else ""
                stage = data.get("stage", "") if data else ""

            # success condition: parseable stage
            if data and stage_to_idx(cfg.task, stage) is not None:
                out.at[i, reason_col] = reasoning
                out.at[i, stage_col] = stage
                out.at[i, raw_col] = raw
                break  # stop retry loop
            # else: retry
        else:
            logger.warning("[REPAIR] Failed to repair row %s after %d attempts.", i, max_retries)

    return out

def extract_ground_truth(df: pd.DataFrame, task: str) -> Optional[pd.Series]:
    if task == "t":
        return df["T14"].astype(int)
    elif task == "n":
        return df["N03"].astype(int)
    return None


def macro_prf(y_true: List[int], y_pred: List[int], num_classes: int) -> Tuple[float, float, float]:
    import numpy as np  # local
    K = num_classes
    tp = np.zeros(K, dtype=int)
    fp = np.zeros(K, dtype=int)
    fn = np.zeros(K, dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yp == yt:
            tp[yt] += 1
        else:
            fp[yp] += 1
            fn[yt] += 1

    def safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    precs = []
    recs = []
    f1s = []
    for k in range(K):
        p = safe_div(tp[k], tp[k] + fp[k])
        r = safe_div(tp[k], tp[k] + fn[k])
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

    return float(sum(precs)) / K, float(sum(recs)) / K, float(sum(f1s)) / K


# ---------------------------
# JSON Schemas via Pydantic
# ---------------------------

class _ResponseRules(BaseModel):  
    rules: List[str] = Field(description="A list of short, explicit rules.")


class _ResponseStageT(BaseModel): 
    reasoning: str = Field(description="Step-by-step reasoning.")
    stage: str = Field(description="Predicted T stage among {T1, T2, T3, T4}.")


class _ResponseStageN(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning.")
    stage: str = Field(description="Predicted N stage among {N0, N1, N2, N3}.")


class _ResponseElicitT(BaseModel): 
    reasoning: str = Field(description="Reasoning for the current report.")
    predicted_stage: str = Field(description="Predicted T stage among {T1, T2, T3, T4}.")
    rules: List[str] = Field(description="Updated rules as a list of short statements.")


class _ResponseElicitN(BaseModel):  
    reasoning: str = Field(description="Reasoning for the current report.")
    predicted_stage: str = Field(description="Predicted N stage among {N0, N1, N2, N3}.")
    rules: List[str] = Field(description="Updated rules as a list of short statements.")


def stage_schema(task: str) -> dict:
    if task == "t":
        return _ResponseStageT.model_json_schema()
    return _ResponseStageN.model_json_schema()     


def elicit_schema(task: str) -> dict:
    if task == "t":
        return _ResponseElicitT.model_json_schema() 
    return _ResponseElicitN.model_json_schema()     


def rules_schema() -> dict:
    return _ResponseRules.model_json_schema()


# ---------------------------
# LLM client
# ---------------------------

class LLMClient:
    def __init__(self, model: str, temperature: float = 0.1) -> None:
        self.client = OpenAI(api_key="dummy_key", base_url="http://127.0.0.1:8000/v1",
                             timeout=httpx.Timeout(300.0, read=300.0, write=120.0, connect=20.0),
                             max_retries=0)
        # assert self.client.models.list().data[0].id == model, f"Model is not set correctly. Expected {model}, got {self.client.models.list().data[0].id}"
        self.model = model
        self.temperature = temperature

    def json_chat(self, messages: List[Dict[str, str]], schema: dict,
                  temperature: Optional[float] = None) -> Tuple[Optional[dict], str]:
        
        temp = self.temperature if temperature is None else temperature
        extra = {"guided_json": schema}
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                extra_body=extra,
            )
        except Exception as e:
            logger.error("[LLM ERROR] %s", e)
            return None, ""

        content = resp.choices[0].message.content or ""
        data = safe_json_load(content)
        if data is not None:
            return data, content

        if schema is not None: # fallback, when data is None
            messages.append({"role": "assistant", "content": content})
            try:
                resp2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages + [{"role": "user", "content": "Return ONLY valid JSON."}],
                    temperature=temp,
                    extra_body=extra,
                )
                content2 = resp2.choices[0].message.content or ""
                data2 = safe_json_load(content2)
                if data2 is not None:
                    return data2, content2
                else:
                    return None, content2
            except Exception as e2:
                logger.error("[LLM ERROR 2] %s", e2)
                return None, content
        return None, content


# ---------------------------
# Prompts
# ---------------------------

SYSTEM_T = "You are an expert in determining the T stage of {cancer_type} cancer following the AJCC Cancer Staging Manual (7th edition)."
SYSTEM_N = "You are an expert in determining the N stage of {cancer_type} cancer following the AJCC Cancer Staging Manual (7th edition)."

PROMPT_ZSCOT = """
You are provided with a pathology report of a {cancer_type} cancer patient.

Pathology report:
{report}

Please:
1) Explain your reasoning step by step as you determine the {task_upper} stage.
2) Provide your final predicted {task_upper} stage ({labels}).
""".strip()

PROMPT_RAG_ONLY = """
You are provided with relevant context retrieved from the AJCC Cancer Staging Manual (7th edition) for {cancer_type} cancer
and a pathology report for a {cancer_type} cancer patient.

Excerpt from AJCC Cancer Staging Manual (7th edition):
{rag_context}

Pathology report:
{report}

Please:
1) Explain your reasoning step by step, integrating this AJCC 7th edition excerpt.
2) Provide your final predicted {task_upper} stage ({labels}).
""".strip()

PROMPT_RULES_ONLY = """
You are provided with official {task_upper} staging rules for {cancer_type} cancer, derived from the AJCC Cancer Staging Manual (7th edition),
and a pathology report for a {cancer_type} cancer patient.

AJCC {task_upper} staging rules (7th edition):
{rules}

Pathology report:
{report}

Please:
1) Explain your reasoning step by step, strictly applying these AJCC 7th edition rules.
2) Provide your final predicted {task_upper} stage ({labels}).
""".strip()

PROMPT_GENERATE_RULES_FROM_RAG = """
You are provided with an excerpt from the AJCC Cancer Staging Manual (7th edition) for {cancer_type} cancer, focusing on {task_upper} stage classification.

Excerpt:
{rag_context}

Please:
1) Summarize the guidelines for {labels} (ignore sub-stages) in bullet form.
2) Return your final summary as a list of short statements that align strictly with the excerpt.
""".strip()

PROMPT_ELICIT_INITIAL = """
You will examine a single pathology report and induce a compact set of decision rules for {task_upper} staging (AJCC 7th).
Then, also predict the stage for this report.

Pathology report:
{report}

Return:
- short step-by-step reasoning,
- a predicted {task_upper} stage ({labels}),
- and a list of short, explicit rules that would help consistently decide {task_upper} in future cases.
""".strip()

PROMPT_ELICIT_UPDATE = """
You are given the current long-term memory (a list of rules) about {task_upper} staging and a new pathology report.
Use the new report to refine (add/remove/rephrase) the rules *only if necessary*, keeping them concise and consistent with AJCC 7th.

Current memory:
{memory}

New pathology report:
{report}

Return:
- short step-by-step reasoning,
- a predicted {task_upper} stage ({labels}),
- and the *updated* list of rules.
""".strip()

PROMPT_INFER_WITH_MEMORY = """
You are provided with long-term memory (rules derived from prior cases) for {task_upper} staging and a new pathology report.

Long-term memory (AJCC 7th-aligned):
{memory}

Pathology report:
{report}

Please:
1) Explain your reasoning step by step (use memory as guard rails; if conflict, AJCC 7th takes priority).
2) Provide your final predicted {task_upper} stage ({labels}).
""".strip()


# ---------------------------
# Config
# ---------------------------

@dataclass
class RunConfig:
    model: str
    task: str               # 't' or 'n'
    cancer_type: str
    temperature: float = 0.1
    seed: int = 42
    context_file: Optional[str] = None
    train_size: int = 40    # for KEwLTM
    edit_threshold: int = 80  # 0..100 fuzzy ratio similarity threshold
    output_csv: Optional[str] = None


# ---------------------------
# Base Method
# ---------------------------

class BaseMethod:
    def __init__(self, llm: LLMClient, cfg: RunConfig) -> None:
        self.llm = llm
        self.cfg = cfg

    @property
    def system_prompt(self) -> str:
        if self.cfg.task == "t":
            return SYSTEM_T.format(cancer_type=self.cfg.cancer_type)
        return SYSTEM_N.format(cancer_type=self.cfg.cancer_type)

    @property
    def label_list(self) -> str:
        return "{T1, T2, T3, T4}" if self.cfg.task == "t" else "{N0, N1, N2, N3}"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


# ---------------------------
# ZSCOT
# ---------------------------

class ZSCOT(BaseMethod):
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        stages: List[Optional[int]] = []
        for idx, row in tqdm(out_df.iterrows(), total=len(out_df), desc="ZSCOT"):
            report = str(row["text"])

            prompt = PROMPT_ZSCOT.format(
                cancer_type=self.cfg.cancer_type,
                report=report,
                task_upper=self.cfg.task.upper(),
                labels=self.label_list,
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            data, raw = self.llm.json_chat(messages, schema=stage_schema(self.cfg.task))
            if not data:
                out_df.loc[idx, "zscot_reasoning"] = ""
                out_df.loc[idx, "zscot_stage"] = ""
                out_df.loc[idx, "zscot_raw_llm"] = raw
                stages.append(None)
                continue
            out_df.loc[idx, "zscot_reasoning"] = data.get("reasoning", "")
            out_df.loc[idx, "zscot_stage"] = data.get("stage", "")
            stages.append(stage_to_idx(self.cfg.task, data.get("stage", "")))

        y_true = extract_ground_truth(out_df, self.cfg.task)
        if y_true is not None and all(s is not None for s in stages):
            y_pred = [int(s) for s in stages]  # type: ignore
            p, r, f1 = macro_prf(list(y_true), y_pred, 4)
            logger.info("[ZSCOT] Macro P=%.3f R=%.3f F1=%.3f", p, r, f1)
        return out_df


# ---------------------------
# Simple RAG
# ---------------------------

class SimpleRAG(BaseMethod):
    def __init__(self, llm: LLMClient, cfg: RunConfig, rag_context: str) -> None:
        super().__init__(llm, cfg)
        self.rag_context = rag_context

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        stages: List[Optional[int]] = []
        for idx, row in tqdm(out_df.iterrows(), total=len(out_df), desc="RAG"):
            report = str(row["text"])
            prompt = PROMPT_RAG_ONLY.format(
                cancer_type=self.cfg.cancer_type,
                rag_context=self.rag_context,
                report=report,
                task_upper=self.cfg.task.upper(),
                labels=self.label_list,
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            data, raw = self.llm.json_chat(messages, schema=stage_schema(self.cfg.task))
            if not data:
                out_df.loc[idx, "rag_reasoning"] = ""
                out_df.loc[idx, "rag_stage"] = ""
                out_df.loc[idx, "rag_raw_llm"] = raw
                stages.append(None)
                continue
            out_df.loc[idx, "rag_reasoning"] = data.get("reasoning", "")
            out_df.loc[idx, "rag_stage"] = data.get("stage", "")
            stages.append(stage_to_idx(self.cfg.task, data.get("stage", "")))

        y_true = extract_ground_truth(out_df, self.cfg.task)
        if y_true is not None and all(s is not None for s in stages):
            y_pred = [int(s) for s in stages]  # type: ignore
            p, r, f1 = macro_prf(list(y_true), y_pred, 4)
            logger.info("[RAG] Macro P=%.3f R=%.3f F1=%.3f", p, r, f1)
        return out_df


# ---------------------------
# KEwRAG
# ---------------------------

class KEwRAG(BaseMethod):
    def __init__(self, llm: LLMClient, cfg: RunConfig, rag_context: str) -> None:
        super().__init__(llm, cfg)
        self.rag_context = rag_context
        self.rules: List[str] = []
        self.rules_raw_fail: Optional[str] = None

    def _generate_rules(self) -> List[str]:
        prompt = PROMPT_GENERATE_RULES_FROM_RAG.format(
            cancer_type=self.cfg.cancer_type,
            rag_context=self.rag_context,
            task_upper=self.cfg.task.upper(),
            labels=self.label_list,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        data, raw = self.llm.json_chat(messages, schema=rules_schema())
        if data and isinstance(data.get("rules"), list):
            rules = [r.strip() for r in data["rules"] if isinstance(r, str) and r.strip()]
            # dedupe
            dedup = []
            seen = set()
            for r in rules:
                k = r.lower()
                if k not in seen:
                    seen.add(k)
                    dedup.append(r)
            return dedup
        # failure: remember raw
        self.rules_raw_fail = raw
        return []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()

        logger.info("[KEwRAG] Generating rules from RAG excerpt...")
        self.rules = self._generate_rules()
        out_df["kewrag_rules"] = [self.rules] * len(out_df)
        if self.rules_raw_fail is not None:
            out_df["kewrag_rules_raw_llm"] = [self.rules_raw_fail] * len(out_df)

        stages: List[Optional[int]] = []
        for idx, row in tqdm(out_df.iterrows(), total=len(out_df), desc="KEwRAG"):
            report = str(row["text"])
            rules_text = "\n".join(self.rules) if self.rules else "No usable rules."
            prompt = PROMPT_RULES_ONLY.format(
                cancer_type=self.cfg.cancer_type,
                rules=rules_text,
                report=report,
                task_upper=self.cfg.task.upper(),
                labels=self.label_list,
            )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            data, raw = self.llm.json_chat(messages, schema=stage_schema(self.cfg.task))
            if not data:
                out_df.loc[idx, "kewrag_reasoning"] = ""
                out_df.loc[idx, "kewrag_stage"] = ""
                out_df.loc[idx, "kewrag_raw_llm"] = raw
                stages.append(None)
                continue
            out_df.loc[idx, "kewrag_reasoning"] = data.get("reasoning", "")
            out_df.loc[idx, "kewrag_stage"] = data.get("stage", "")
            stages.append(stage_to_idx(self.cfg.task, data.get("stage", "")))

        y_true = extract_ground_truth(out_df, self.cfg.task)
        if y_true is not None and all(s is not None for s in stages):
            y_pred = [int(s) for s in stages]  # type: ignore
            p, r, f1 = macro_prf(list(y_true), y_pred, 4)
            logger.info("[KEwRAG] Macro P=%.3f R=%.3f F1=%.3f", p, r, f1)
        return out_df


# ---------------------------
# KEwLTM
# ---------------------------

class KEwLTM(BaseMethod):
    def __init__(self, llm: LLMClient, cfg: RunConfig) -> None:
        super().__init__(llm, cfg)
        self.memory: List[str] = []

    def _accept_update(self, old: List[str], new: List[str]) -> bool:
        if not old:
            return True
        if not new:
            return False
        if fuzz is None:
            return 0.6 <= (len(set(new)) / max(1, len(set(old))))
        old_str = "\n".join(old)
        new_str = "\n".join(new)
        ratio = fuzz.ratio(old_str, new_str)
        return ratio >= self.cfg.edit_threshold

    def _normalize_rules(self, rules: List[str]) -> List[str]:
        cleaned = []
        seen = set()
        for r in rules:
            if not isinstance(r, str):
                continue
            s = r.strip()
            if not s:
                continue
            k = re.sub(r"\s+", " ", s.lower())
            if k in seen:
                continue
            seen.add(k)
            cleaned.append(s)
        return cleaned

    def _elicit_initial(self, report: str) -> Tuple[List[str], str, str, str]:
        prompt = PROMPT_ELICIT_INITIAL.format(
            report=report,
            task_upper=self.cfg.task.upper(),
            labels=self.label_list,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        data, raw = self.llm.json_chat(messages, schema=elicit_schema(self.cfg.task))
        if not data:
            return [], "", "", raw
        rules = self._normalize_rules(data.get("rules", []) or [])
        return rules, data.get("reasoning", ""), data.get("predicted_stage", ""), raw

    def _elicit_update(self, report: str, memory: List[str]) -> Tuple[List[str], str, str, str]:
        prompt = PROMPT_ELICIT_UPDATE.format(
            memory="\n".join(memory) if memory else "(empty)",
            report=report,
            task_upper=self.cfg.task.upper(),
            labels=self.label_list,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        data, raw = self.llm.json_chat(messages, schema=elicit_schema(self.cfg.task))
        if not data:
            return memory, "", "", raw
        rules = self._normalize_rules(data.get("rules", []) or [])
        return rules, data.get("reasoning", ""), data.get("predicted_stage", ""), raw

    def _infer_with_memory(self, report: str, memory: List[str]) -> Tuple[str, str, str]:
        prompt = PROMPT_INFER_WITH_MEMORY.format(
            memory="\n".join(memory) if memory else "(empty)",
            report=report,
            task_upper=self.cfg.task.upper(),
            labels=self.label_list,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        data, raw = self.llm.json_chat(messages, schema=stage_schema(self.cfg.task))
        if not data:
            return "", "", raw
        return data.get("reasoning", ""), data.get("stage", ""), raw

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        train_size = min(self.cfg.train_size, len(out_df))

        # --- Initialize all output columns with empty values beforehand ---
        out_df["kewltm_reasoning"] = None
        out_df["kewltm_stage"] = None
        out_df["kewltm_raw_llm"] = None
        out_df["kewltm_train_raw_llm"] = None
        out_df["kewltm_memory_snapshot"] = None
        out_df["kewltm_memory_snapshot_str"] = None
        out_df["is_train"] = False # Flag to mark training rows

        logger.info("[KEwLTM] Inducing memory from %d reports (threshold=%d)...",
                    train_size, self.cfg.edit_threshold)
        
        stages_for_eval: List[Optional[int]] = []

        # --- A single loop for both training and inference ---
        for idx, row in tqdm(out_df.iterrows(), total=len(out_df), desc="KEwLTM"):
            report = str(row["text"])

            # ---- TRAINING PHASE ----
            if idx < train_size:
                out_df.at[idx, "is_train"] = True
                
                if not self.memory: # First training instance
                    new_rules, _, _, raw = self._elicit_initial(report)
                    if new_rules:
                        self.memory = new_rules
                    else:
                        out_df.loc[idx, "kewltm_train_raw_llm"] = raw
                else: # Update existing memory
                    updated_rules, _, _, raw = self._elicit_update(report, self.memory)
                    if updated_rules == self.memory:
                        if not updated_rules and raw:
                            out_df.loc[idx, "kewltm_train_raw_llm"] = raw
                    elif self._accept_update(self.memory, updated_rules):
                        self.memory = updated_rules

                out_df.at[idx, "kewltm_memory_snapshot"] = list(self.memory)
                out_df.at[idx, "kewltm_memory_snapshot_str"] = "\n".join(self.memory)

            # ---- INFERENCE PHASE ----
            else:
                reasoning, stage, raw = self._infer_with_memory(report, self.memory)
                
                if not stage:
                    out_df.loc[idx, "kewltm_raw_llm"] = raw
                
                out_df.loc[idx, "kewltm_reasoning"] = reasoning
                out_df.loc[idx, "kewltm_stage"] = stage
                stages_for_eval.append(stage_to_idx(self.cfg.task, stage) if stage else None)

        logger.info("[KEwLTM] Final memory length: %d", len(self.memory))
        out_df["kewltm_memory"] = [self.memory] * len(out_df)

        # --- Evaluation ---
        # Note: We create a boolean mask to select only the test rows for evaluation
        test_mask = ~out_df["is_train"]
        y_true_test = extract_ground_truth(out_df[test_mask], self.cfg.task)
        
        if y_true_test is not None and all(s is not None for s in stages_for_eval):
            y_pred = [int(s) for s in stages_for_eval]
            p, r, f1 = macro_prf(list(y_true_test), y_pred, 4)
            logger.info("[KEwLTM] Macro P=%.3f R=%.3f F1=%.3f", p, r, f1)

        return out_df


# ---------------------------
# IO helpers
# ---------------------------

def load_context(context_file: str, task: str) -> str:
    with open(context_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = "rag_raw_t14" if task == "t" else "rag_raw_n03"
    if key not in data:
        raise KeyError(f"Context JSON is missing key '{key}'")
    return str(data[key])


def read_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    return df


def write_output(df: pd.DataFrame, out_path: Optional[str]) -> None:
    if not out_path:
        out_path = "results.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved results to %s", out_path)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run KEwLTM, KEwRAG, RAG, or ZSCOT on a dataset (with logging).")
    parser.add_argument("--method", required=True, choices=["zscot", "rag", "kewrag", "kewltm"])
    parser.add_argument("--task", required=True, choices=["t", "n"], help="t=T1..T4, n=N0..N3")
    parser.add_argument("--dataset", required=True, help="Path to CSV with a 'text' column.")
    parser.add_argument("--context-file", default=None, help="JSON with 'rag_raw_t14'/'rag_raw_n03' (for rag/kewrag).")
    parser.add_argument("--cancer-type", default="breast", help="Human-readable cancer type for prompts.")
    parser.add_argument("--model", required=True, help="Model name served by your OpenAI-compatible endpoint.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=40, help="KEwLTM: number of reports to induce memory.")
    parser.add_argument("--edit-threshold", type=int, default=80, help="KEwLTM: fuzzy ratio threshold [0..100].")
    parser.add_argument("--output-csv", default=None, help="Path to save output CSV.")
    parser.add_argument("--log-file", default=None, help="Optional path to a log file.")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR, CRITICAL")
    parser.add_argument("--repair-input", default=None,
                        help="Path to an EXISTING results CSV to repair in-place (only missing {method}_stage will be re-predicted).")
    parser.add_argument("--repair-output-dir", default=None,
                        help="Used with --repair-input. If set, write repaired CSV into this directory keeping the same basename (unless --output-csv is provided).")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max attempts per row during repair.")
    parser.add_argument("--treat-unparseable-as-missing", action="store_true",
                        help="Also repair rows whose stage string exists but cannot be parsed (e.g., 'stage ?').")

    args = parser.parse_args()

    setup_logging(args.log_file, args.log_level)

    # Seed
    np.random.seed(args.seed)


    cfg = RunConfig(
        model=args.model,
        task=args.task,
        cancer_type=args.cancer_type,
        temperature=args.temperature,
        seed=args.seed,
        context_file=args.context_file,
        train_size=args.train_size,
        edit_threshold=args.edit_threshold,
        output_csv=args.output_csv,
    )

    # LLM
    llm = LLMClient(model=cfg.model, temperature=cfg.temperature)

    # If --repair-input is provided, run the repair pass and exit.
    if args.repair_input:
        setup_logging(args.log_file, args.log_level)
        llm = LLMClient(model=args.model, temperature=args.temperature)
        cfg = RunConfig(
            model=args.model,
            task=args.task,
            cancer_type=args.cancer_type,
            temperature=args.temperature,
            seed=args.seed,
            context_file=args.context_file,
            train_size=args.train_size,
            edit_threshold=args.edit_threshold,
            output_csv=args.output_csv,
        )
        df_in = pd.read_csv(args.repair_input)

        rag_ctx = None
        if args.method in {"rag", "kewrag"} and args.context_file:
            rag_ctx = load_context(args.context_file, args.task)

        df_out = repair_missing_predictions(
            method=args.method,
            llm=llm,
            cfg=cfg,
            df=df_in,
            max_retries=args.max_retries,
            treat_unparseable_as_missing=args.treat_unparseable_as_missing,
            rag_context=rag_ctx,
        )

        # # default output path if not provided
        # out_path = args.output_csv or re.sub(r"\.csv$", "", args.repair_input) + "__repair.csv"
        # write_output(df_out, out_path)
        # return
        # default output path if not provided
        if args.output_csv:
            out_path = args.output_csv
        elif args.repair_output_dir:
            os.makedirs(args.repair_output_dir, exist_ok=True)
            out_path = str(Path(args.repair_input).with_name(Path(args.repair_input).name)
                           .with_suffix(".csv"))
            out_path = str(Path(args.repair_output_dir) / Path(out_path).name)  # <-- 같은 파일명, 다른 디렉토리
        else:
            out_path = re.sub(r"\.csv$", "", args.repair_input) + "__repair.csv"

        write_output(df_out, out_path)
        return


    # Data
    df = read_dataset(args.dataset).copy()
    if args.method == "kewltm":
        df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    if args.method == "zscot":
        runner = ZSCOT(llm, cfg)
        out = runner.run(df)
    elif args.method == "rag":
        if not cfg.context_file:
            raise ValueError("--context-file is required for method=rag")
        rag_ctx = load_context(cfg.context_file, cfg.task)
        runner = SimpleRAG(llm, cfg, rag_context=rag_ctx)
        out = runner.run(df)
    elif args.method == "kewrag":
        if not cfg.context_file:
            raise ValueError("--context-file is required for method=kewrag")
        rag_ctx = load_context(cfg.context_file, cfg.task)
        runner = KEwRAG(llm, cfg, rag_context=rag_ctx)
        out = runner.run(df)
    else:  # kewltm
        runner = KEwLTM(llm, cfg)
        out = runner.run(df)

    write_output(out, cfg.output_csv)


if __name__ == "__main__":
    main()
