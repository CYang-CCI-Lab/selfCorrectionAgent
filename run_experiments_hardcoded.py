#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments_hardcoded.py

Batch runner that uses hard-coded configuration (edit the CONFIG section below)
to run ZSCOT, RAG, KEwRAG, and/or KEwLTM across per-cancer CSVs with matching
context JSON files. No command-line arguments required.

It imports the classes and helpers you already have in kew_methods.py and utils.py.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import your existing modules (assumed to be on PYTHONPATH or same folder)
import kew_methods as km
from utils import setup_logging, get_logger

logger = get_logger()

# ======================================================================================
# CONFIG — EDIT THESE VALUES
# ======================================================================================
CONFIG = {
    # Root of your project (folder that contains per_cancer_type/ and rag/context/)
    "DATA_ROOT": "/home/yl3427/cylab/selfCorrectionAgent",

    # Subfolders relative to DATA_ROOT
    "PER_CANCER_DIR": "per_cancer_type",
    "CONTEXT_DIR": "rag/context",

    # Where to write CSV results and logs
    "OUT_DIR": "runs2",

    # Model to use. If you leave this as None, the script will query the first model id
    # from your vLLM/OpenAI-compatible server at http://localhost:8000/v1
    "MODEL":"mistralai/Mixtral-8x7B-Instruct-v0.1",  

    # Which methods to run (any subset). Valid options: zscot, rag, kewrag, kewltm
    "METHODS": ["zscot", "rag", "kewrag", "kewltm"], # "rag", "kewrag", "kewltm"

    # Limit to only these TCGA codes (or set to [] to include all)
    "ONLY": [],  # e.g., ["BRCA", "LUAD"]

    # Skip these TCGA codes
    "SKIP": [],  # e.g., ["KIRC"]

    # Global RNG seed
    "SEED": 42,

    # === Dynamic train-size control for KEwLTM ===
    # Use ~5% of the number of rows available for the task (T14 or N03) in each cancer type.
    # If you want to force a fixed integer instead, set FORCE_TRAIN_SIZE to a positive int.
    "TRAIN_FRACTION": 0.1,     # 0.05 -> 5% (matches 40/800 in your BRCA setup)
    "FORCE_TRAIN_SIZE": 5,   # e.g., set to 40 to override dynamic sizing

    # KEwLTM similarity gate
    "EDIT_THRESHOLD": 80,

    # Log level for per-run logs: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "LOG_LEVEL": "INFO",
}
# ======================================================================================

# Mapping for prompts (TCGA code -> printable name)
TCGA_TO_PRINTABLE: Dict[str, str] = {
    "BLCA": "Bladder Urothelial Carcinoma",
    "HNSC": "Head and Neck Squamous Cell Carcinoma",
    "STAD": "Stomach Adenocarcinoma",
    "CESC": "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma",
    "KIRC": "Kidney Renal Clear Cell Carcinoma",
    "PRAD": "Prostate Adenocarcinoma",
    "KIRP": "Kidney Renal Papillary Cell Carcinoma",
    "KICH": "Kidney Chromophobe",
    "LIHC": "Liver Hepatocellular Carcinoma",
    "BRCA": "Breast Invasive Carcinoma",
    "LUAD": "Lung Adenocarcinoma",
    "PAAD": "Pancreatic Adenocarcinoma",
    "THCA": "Thyroid Carcinoma",
    "MESO": "Mesothelioma",
    "ACC":  "Adrenocortical Carcinoma",
    "CHOL": "Cholangiocarcinoma",
    "TGCT": "Testicular Germ Cell Tumors",
    "LUSC": "Lung Squamous Cell Carcinoma",
    "READ": "Rectum Adenocarcinoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "COAD": "Colon Adenocarcinoma",
    "UVM":  "Uveal Melanoma",
    "ESCA": "Esophageal Carcinoma",
}

SUPPORTED_METHODS = {"zscot", "rag", "kewrag", "kewltm"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_per_cancer_csvs(per_cancer_dir: Path) -> List[Path]:
    return sorted(per_cancer_dir.glob("*_T14N03.csv"))


def tcga_from_filename(path: Path) -> Optional[str]:
    m = re.match(r"([A-Za-z]+)_T14N03\.csv$", path.name)
    return m.group(1).upper() if m else None


def context_path_for(tcga_code: str, context_dir: Path) -> Path:
    return context_dir / f"context_{tcga_code}.json"


def has_column(csv_path: Path, colname: str) -> bool:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return colname in header


def discover_model_id() -> str:
    """
    Use the same OpenAI-compatible settings as in kew_methods.LLMClient to read
    the first available model id so it matches the assertion inside LLMClient.
    """
    from openai import OpenAI
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1")
    return client.models.list().data[0].id


def effective_task_rows(df, task: str) -> int:
    """
    Count rows that have a label for the task, if available.
    Falls back to total rows if the label column is absent.
    This does not filter the dataset used for induction; it is
    used only to choose a reasonable train_size proportionally.
    """
    if task == "t" and "T14" in df.columns:
        return int(df["T14"].notna().sum())
    if task == "n" and "N03" in df.columns:
        return int(df["N03"].notna().sum())
    return int(len(df))


def choose_dynamic_train_size(df, task: str) -> int:
    """
    Compute train_size as approximately TRAIN_FRACTION of the number of rows
    available for the task. Always at least 1 and at most len(df).
    """
    if isinstance(CONFIG.get("FORCE_TRAIN_SIZE"), int) and CONFIG["FORCE_TRAIN_SIZE"] > 0:
        return min(CONFIG["FORCE_TRAIN_SIZE"], len(df))

    frac = float(CONFIG.get("TRAIN_FRACTION", 0.05))
    n_eff = effective_task_rows(df, task)
    size = max(1, int(round(n_eff * frac)))
    size = min(size, len(df))
    return size


def run_one_method(
    method: str,
    task: str,
    dataset: Path,
    context_file: Optional[Path],
    model: str,
    cancer_type_printable: str,
    seed: int,
    edit_threshold: int,
    out_csv: Path,
    log_file: Path,
    log_level: str,
) -> None:
    """
    Run a single (method, task) on one dataset and save results to out_csv.
    Reconfigures the shared logger to write into log_file for this run.
    """
    setup_logging(str(log_file), log_level)  # reconfigure shared logger
    logger.info("=== RUN START === method=%s task=%s dataset=%s model=%s", method, task, dataset, model)

    # Load data
    df = km.read_dataset(str(dataset))

    # Filter the DataFrame to only include rows with valid ground truth for the task
    gt_col = None
    if task == "t":
        gt_col = "T14" 
    elif task == "n":
        gt_col = "N03" 

    original_rows = len(df)
    df = df[df[gt_col].notna()].copy().reset_index(drop=True)
    filtered_rows = len(df)
    if original_rows > filtered_rows:
        logger.info(
            "Filtered dataset: Removed %d rows with missing '%s' labels.",
            original_rows - filtered_rows, gt_col
        )

    # For KEwLTM, compute a dynamic train size based on the *task* rows
    if method == "kewltm":
        train_size = choose_dynamic_train_size(df, task)
        logger.info("[KEwLTM] Dynamic train_size=%d (fraction=%.3f)", train_size, CONFIG["TRAIN_FRACTION"])
        # Shuffle so the first `train_size` rows form the memory set
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        train_size = 0  # unused

    # Build config
    cfg = km.RunConfig(
        model=model,
        task=task,
        cancer_type=cancer_type_printable,
        temperature=0.1,
        seed=seed,
        context_file=str(context_file) if context_file else None,
        train_size=train_size if method == "kewltm" else 0,  # only used by KEwLTM
        edit_threshold=edit_threshold,
        output_csv=str(out_csv),
    )
    # LLM client
    llm = km.LLMClient(model=cfg.model, temperature=cfg.temperature)

    # Runner selection
    if method == "zscot":
        runner = km.ZSCOT(llm, cfg)
        out = runner.run(df)
    elif method == "rag":
        if not cfg.context_file:
            raise FileNotFoundError("RAG requires a context JSON path.")
        rag_ctx = km.load_context(cfg.context_file, cfg.task)
        runner = km.SimpleRAG(llm, cfg, rag_context=rag_ctx)
        out = runner.run(df)
    elif method == "kewrag":
        if not cfg.context_file:
            raise FileNotFoundError("KEwRAG requires a context JSON path.")
        rag_ctx = km.load_context(cfg.context_file, cfg.task)
        runner = km.KEwRAG(llm, cfg, rag_context=rag_ctx)
        out = runner.run(df)
    elif method == "kewltm":
        runner = km.KEwLTM(llm, cfg)
        out = runner.run(df)
    else:
        raise ValueError(f"Unsupported method: {method}")

    km.write_output(out, cfg.output_csv)
    logger.info("=== RUN END === results=%s", out_csv)


def run_all() -> None:
    # Resolve paths
    root = Path(CONFIG["DATA_ROOT"]).resolve()
    per_cancer_dir = (root / CONFIG["PER_CANCER_DIR"]).resolve()
    context_dir = (root / CONFIG["CONTEXT_DIR"]).resolve()
    out_dir = (root / CONFIG["OUT_DIR"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine model id (if not specified)
    model_id = CONFIG["MODEL"] or discover_model_id()

    # Validate methods
    methods = [m.lower() for m in CONFIG["METHODS"]]
    for m in methods:
        if m not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {m} (choose from {sorted(SUPPORTED_METHODS)})")

    # Find datasets
    csvs = find_per_cancer_csvs(per_cancer_dir)
    if not csvs:
        raise FileNotFoundError(f"No *_T14N03.csv files found in {per_cancer_dir}")

    only = set([x.upper() for x in (CONFIG.get("ONLY") or [])])
    skip = set([x.upper() for x in (CONFIG.get("SKIP") or [])])

    # Manifest
    manifest_path = out_dir / f"manifest__{timestamp()}.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as mf:
        mw = csv.writer(mf)
        mw.writerow([
            "tcga", "printable", "dataset",
            "task", "method", "context_json",
            "model", "seed", "train_fraction", "computed_train_size", "edit_threshold",
            "results_csv", "log_file"
        ])

        for csv_path in csvs:
            tcga = tcga_from_filename(csv_path)
            if not tcga:
                print(f"[skip] Could not parse TCGA code from {csv_path.name}")
                continue
            if only and tcga not in only:
                continue
            if tcga in skip:
                continue

            printable = TCGA_TO_PRINTABLE.get(tcga, tcga)
            context_json = context_path_for(tcga, context_dir)

            # Which tasks?
            has_t = has_column(csv_path, "T14")
            has_n = has_column(csv_path, "N03")
            tasks = []
            if has_t:
                tasks.append("t")
            if has_n:
                tasks.append("n")
            if not tasks:
                print(f"[skip] {csv_path.name} has neither T14 nor N03.")
                continue

            # Context availability for RAG methods
            ctx_for_methods = context_json if context_json.exists() else None

            # Load once for computing dynamic train size per task
            df_for_counts = km.read_dataset(str(csv_path))

            for task in tasks:
                # Compute anticipated dynamic train size (only meaningful for KEwLTM)
                anticipated_train = choose_dynamic_train_size(df_for_counts, task)

                for method in methods:
                    if method in {"rag", "kewrag"} and ctx_for_methods is None:
                        print(f"[warn] Missing context for {tcga}. Skipping {method} ({task}).")
                        continue

                    base = (
                        f"{csv_path.stem}__{method}__{task}"
                        f"__{model_id.replace('/', '_')}"
                        f"__seed{CONFIG['SEED']}"
                        f"__{timestamp()}"
                    )
                    out_csv = out_dir / f"{base}.csv"
                    log_file = out_dir / f"{base}.log"

                    try:
                        run_one_method(
                            method=method,
                            task=task,
                            dataset=csv_path,
                            context_file=ctx_for_methods,
                            model=model_id,
                            cancer_type_printable=printable,
                            seed=CONFIG["SEED"],
                            edit_threshold=CONFIG["EDIT_THRESHOLD"],
                            out_csv=out_csv,
                            log_file=log_file,
                            log_level=CONFIG["LOG_LEVEL"],
                        )
                    except Exception as e:
                        # Reconfigure to print to console as well
                        setup_logging(None, CONFIG["LOG_LEVEL"])
                        logger.exception("Run failed for %s %s on %s: %s", method, task, csv_path.name, e)

                    mw.writerow([
                        tcga, printable, str(csv_path),
                        task, method, str(ctx_for_methods) if ctx_for_methods else "",
                        model_id, CONFIG["SEED"], CONFIG["TRAIN_FRACTION"], anticipated_train, CONFIG["EDIT_THRESHOLD"],
                        str(out_csv), str(log_file)
                    ])
                    mf.flush()

    print(f"\nAll done. Manifest: {manifest_path}\nResults/logs: {out_dir}\n")


if __name__ == "__main__":
    run_all()
