from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

import kew_methods as km
from utils import setup_logging, get_logger

logger = get_logger()
PAT_RESULT = re.compile(r"__(?P<method>kewltm|kewrag|rag|zscot)__(?P<task>[tn])__")
# ======================================================================================
# CONFIG
# ======================================================================================
CONFIG = {
    "DATA_ROOT": "/home/yl3427/cylab/selfCorrectionAgent",
    "PER_CANCER_DIR": "per_cancer_type",
    "CONTEXT_DIR": "rag/context",
    "OUT_DIR": "runs_med",
    "MODEL":"m42-health/Llama3-Med42-70B",  # "m42-health/Llama3-Med42-70B", "mistralai/Mixtral-8x7B-Instruct-v0.1"
    "METHODS": ["kewltm", "zscot", "rag", "kewrag"], # "rag", "kewrag", "kewltm"
    "ONLY": [],  # ["BRCA", "LUAD"]
    "SKIP": [],  # ["KIRC"]
    "SEED": 42,

    # If you want to force a fixed integer instead, set FORCE_TRAIN_SIZE to a positive int.
    "TRAIN_FRACTION": 0.05,     # 0.05 -> 5% (matches 40/800 in your BRCA setup)
    "FORCE_TRAIN_SIZE": None,   # e.g., set to 40 to override dynamic sizing

    "EDIT_THRESHOLD": 80,
    "LOG_LEVEL": "INFO",


    # ====== 리페어 모드 관련 ======
    "REPAIR_ONLY": False,               # True면 "새 실험"은 건너뛰고 리페어만 수행
    "REPAIR_AFTER_RUN": True,          # 새 실험(run_all)을 돌린 직후에 리페어도 자동으로 돌리고 싶을 때만 True로 둡니다. REPAIR_ONLY와는 동시에 True로 쓰지 마세요. (둘 중 하나)


    "REPAIR_INPUT_DIR": "runs_med",         # 리페어 대상 파일들이 있는 디렉토리

    "REPAIR_MAX_RETRIES": 2,            # 행당 재시도 횟수
    "REPAIR_TREAT_UNPARSEABLE_AS_MISSING": True,  # "stage 문자열은 있지만 파싱 불가"도 결측으로 간주
    "REPAIR_METHOD_FILTER": [],          # 예: ["kewltm", "rag"] 지정하면 그 메서드 결과만 리페어. 빈 리스트면 전부

    # ===== 리페어 디렉토리/반복 설정 =====
    "REPAIR_OUTPUT_DIR": "runs_med2",                 # 1회 리페어 출력 디렉토리 (파일명 유지)
    "REPAIR_OUTPUT_DIR_PATTERN": "runs{}",        # 반복 리페어용 디렉토리 패턴
    "REPAIR_FIRST_INDEX": 2,                      # runs2부터 시작
    "REPAIR_ITERATE_ROUNDS": 1,                   # 1이면 한 번, 2 이상이면 반복

    "REPAIR_STOP_EARLY": False,                    # 라운드에서 수정된 행이 0이면 조기 종료
    "REPAIR_PATIENCE": 1,         # <- 연속 '무개선' 라운드 허용 횟수 (0이면 즉시 중단, 1이면 2연속 무개선 시 중단)

    "REPAIR_KEEP_SAME_FILENAME": True,            # 출력 파일명 유지 (True 권장)


    # ---- repair 전용 필터/재개 옵션 ----
    "REPAIR_ONLY_TCGA": [],                 # ex) ["BRCA", "LUAD"]  (미설정/빈 리스트면 아래 ONLY를 fallback)
    "REPAIR_SKIP_TCGA": [],                 # ex) ["KIRC"]          (미설정/빈 리스트면 아래 SKIP을 fallback)
    "REPAIR_SKIP_IF_OUTPUT_EXISTS": True,   # runs2에 동일 파일명이 있으면 건너뛰기(재개용)
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
    from openai import OpenAI
    client = OpenAI(api_key="dummy_key", base_url="http://localhost:8000/v1")
    return client.models.list().data[0].id


def effective_task_rows(df, task: str) -> int:
    if task == "t" and "T14" in df.columns:
        return int(df["T14"].notna().sum())
    if task == "n" and "N03" in df.columns:
        return int(df["N03"].notna().sum())
    return int(len(df))


def choose_dynamic_train_size(df, task: str) -> int:
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

    if method == "kewltm":
        train_size = choose_dynamic_train_size(df, task)
        logger.info("[KEwLTM] Dynamic train_size=%d (fraction=%.3f)", train_size, CONFIG["TRAIN_FRACTION"])
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        train_size = 0  # unused

    cfg = km.RunConfig(
        model=model,
        task=task,
        cancer_type=cancer_type_printable,
        temperature=0.2,
        seed=seed,
        context_file=str(context_file) if context_file else None,
        train_size=train_size if method == "kewltm" else 0,  # only used by KEwLTM
        edit_threshold=edit_threshold,
        output_csv=str(out_csv),
    )

    llm = km.LLMClient(model=cfg.model, temperature=cfg.temperature)

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
    root = Path(CONFIG["DATA_ROOT"]).resolve()
    per_cancer_dir = (root / CONFIG["PER_CANCER_DIR"]).resolve()
    context_dir = (root / CONFIG["CONTEXT_DIR"]).resolve()
    out_dir = (root / CONFIG["OUT_DIR"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = CONFIG["MODEL"] or discover_model_id()

    methods = [m.lower() for m in CONFIG["METHODS"]]
    for m in methods:
        if m not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {m} (choose from {sorted(SUPPORTED_METHODS)})")
        
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
                logger.info(f"[skip] Could not parse TCGA code from {csv_path.name}")
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
                logger.info(f"[skip] {csv_path.name} has neither T14 nor N03.")
                continue

            ctx_for_methods = context_json if context_json.exists() else None

            df_for_counts = km.read_dataset(str(csv_path))

            for task in tasks:
                anticipated_train = choose_dynamic_train_size(df_for_counts, task)

                for method in methods:
                    if method in {"rag", "kewrag"} and ctx_for_methods is None:
                        logger.info(f"[warn] Missing context for {tcga}. Skipping {method} ({task}).")
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
                        logger.exception("Run failed for %s %s on %s: %s", method, task, csv_path.name, e)

                    mw.writerow([
                        tcga, printable, str(csv_path),
                        task, method, str(ctx_for_methods) if ctx_for_methods else "",
                        model_id, CONFIG["SEED"], CONFIG["TRAIN_FRACTION"], anticipated_train, CONFIG["EDIT_THRESHOLD"],
                        str(out_csv), str(log_file)
                    ])
                    mf.flush()

    logger.info(f"\nAll done. Manifest: {manifest_path}\nResults/logs: {out_dir}\n")

def parse_method_task_from_results(path: Path) -> Optional[Tuple[str, str]]:
    m = PAT_RESULT.search(path.name)
    if not m:
        return None
    return m.group("method"), m.group("task")

def count_missing_for_method(series: pd.Series, task: str, *, treat_unparseable: bool) -> int:
    # 결측: NaN 또는 공백 문자열
    missing = series.isna() | series.astype(str).str.strip().eq("")
    if treat_unparseable:
        # 문자열은 있으나 stage_to_idx 파싱 실패도 결측으로 간주
        mask = []
        for v in series.fillna(""):
            if isinstance(v, str) and v.strip() != "":
                mask.append(km.stage_to_idx(task, v) is None)
            else:
                mask.append(False)
        missing |= pd.Series(mask, index=series.index)
    return int(missing.sum())

def count_missing_for_method_in_df(
    df: pd.DataFrame, method: str, task: str, *, treat_unparseable: bool
) -> int:
    s = df[f"{method}_stage"]
    # KEwLTM은 학습행은 리페어 대상에서 제외되므로 결측 계산에서도 제외
    if method == "kewltm" and "is_train" in df.columns:
        mask = ~df["is_train"].fillna(False)
        s = s[mask]
    return count_missing_for_method(s, task, treat_unparseable=treat_unparseable)


def run_repairs(input_dir: Path, output_dir: Path, *, round_tag: str = "") -> int:
    """
    input_dir의 결과 CSV들을 배치 리페어해서 output_dir에 같은 파일명으로 저장.
    반환값: 이번 라운드에서 '수리된 행 수'의 총합 (조기종료 판단에 사용)
    """
    root = Path(CONFIG["DATA_ROOT"]).resolve()
    context_dir = (root / CONFIG["CONTEXT_DIR"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"repair{('_' + round_tag if round_tag else '')}.log"
    setup_logging(str(log_file), CONFIG["LOG_LEVEL"])

    model_id = CONFIG["MODEL"] or discover_model_id()
    shared_llm = km.LLMClient(model=model_id)

    # === NEW: repair용 TCGA 필터 세트 준비 (repair 전용 키가 비면 기존 ONLY/SKIP fallback) ===
    only_tcga = set(x.upper() for x in (CONFIG.get("REPAIR_ONLY_TCGA") or CONFIG.get("ONLY") or []))
    skip_tcga = set(x.upper() for x in (CONFIG.get("REPAIR_SKIP_TCGA") or CONFIG.get("SKIP") or []))
    skip_if_exists = bool(CONFIG.get("REPAIR_SKIP_IF_OUTPUT_EXISTS", True))

    # 라운드별 매니페스트는 출력 디렉토리에 저장
    manifest_path = output_dir / f"manifest_repair{('_' + round_tag if round_tag else '')}__{timestamp()}.csv"
    total_repaired = 0

    with manifest_path.open("w", encoding="utf-8", newline="") as mf:
        mw = csv.writer(mf)
        mw.writerow([
            "tcga", "printable", "input_csv", "output_csv",
            "method", "task", "model", "seed",
            "missing_before", "missing_after", "repaired", "log"
        ])

        for p in sorted(input_dir.glob("*.csv")):
            if p.name.startswith("manifest"):  # 매니페스트들은 건너뜀
                continue

            mt = parse_method_task_from_results(p)
            if not mt:
                continue
            method, task = mt
            if CONFIG["REPAIR_METHOD_FILTER"] and method not in CONFIG["REPAIR_METHOD_FILTER"]:
                continue

            tcga = p.stem.split("_")[0].upper()
            printable = TCGA_TO_PRINTABLE.get(tcga, tcga)
    
            # === NEW: this file's output path (필요 시 스킵 판단 위해 '먼저' 계산) ===
            out_p = (output_dir / p.name) if CONFIG["REPAIR_KEEP_SAME_FILENAME"] else (output_dir / (p.stem + "__repair.csv"))

            # === NEW: TCGA 필터 적용 (repair 전용 -> 없으면 기존 ONLY/SKIP fallback) ===
            if only_tcga and tcga not in only_tcga:
                # 매니페스트에 기록(선택)
                mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
                            "", "", "", "skip_only_filter"])
                mf.flush()
                continue
            if tcga in skip_tcga:
                mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
                            "", "", "", "skip_skip_filter"])
                mf.flush()
                continue

            # === NEW: 재개 모드 - 이미 출력이 있으면 스킵 ===
            # if skip_if_exists and out_p.exists():
            #     logger.info("[REPAIR] %s: output exists -> skip (%s)", p.name, out_p.name)
            #     mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
            #                 "", "", "", "skip_existing_output"])
            #     mf.flush()
            #     continue
            if skip_if_exists and out_p.exists():
                # 입력이 더 새로우면 다시 돌림, 아니면 스킵
                try:
                    if out_p.stat().st_mtime >= p.stat().st_mtime:
                        logger.info("[REPAIR] %s: output exists and is up-to-date -> skip (%s)", p.name, out_p.name)
                        mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
                                    "", "", "", "skip_existing_output"])
                        mf.flush()
                        continue
                except Exception as e:
                    logger.warning("[REPAIR] mtime check failed for %s -> proceed: %s", p.name, e)


            cfg = km.RunConfig(
                model=model_id,
                task=task,
                cancer_type=printable,
                temperature=0.2,
                seed=CONFIG["SEED"],
                context_file=str(context_dir / f"context_{tcga}.json") if (context_dir / f"context_{tcga}.json").exists() else None,
                train_size=0,
                edit_threshold=CONFIG["EDIT_THRESHOLD"],
                output_csv=None,
            )
            # llm = km.LLMClient(model=cfg.model, temperature=cfg.temperature)

            try:
                df_in = pd.read_csv(p)
            except Exception as e:
                logger.exception("[REPAIR] Failed to read %s: %s", p, e)
                mw.writerow([tcga, printable, str(p), "", method, task, model_id, CONFIG["SEED"], "", "", "", f"read_error: {e}"])
                mf.flush()
                continue

            stage_col = f"{method}_stage"
            if stage_col not in df_in.columns:
                logger.info("[REPAIR] %s: missing '%s' column; skip.", p.name, stage_col)
                continue

            # missing_before = count_missing_for_method(
            #     df_in[stage_col], task, treat_unparseable=CONFIG["REPAIR_TREAT_UNPARSEABLE_AS_MISSING"]
            # )
            missing_before = count_missing_for_method_in_df(
                df_in, method, task, treat_unparseable=CONFIG["REPAIR_TREAT_UNPARSEABLE_AS_MISSING"]
            )
            if missing_before == 0:
                # out_p = output_dir / p.name if CONFIG["REPAIR_KEEP_SAME_FILENAME"] else (output_dir / (p.stem + "__repair.csv"))
                # 굳이 복사/저장을 안 해도 되지만, 파이프라인 일관성을 위해 저장하는 게 안전
                try:
                    df_in.to_csv(out_p, index=False)
                except Exception as e:
                    logger.warning("[REPAIR] %s: nothing to repair but writing failed: %s", p.name, e)
                mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
                             0, 0, 0, "no_missing"])
                mf.flush()
                continue

            rag_ctx = None
            if method in {"rag", "kewrag"} and cfg.context_file:
                try:
                    rag_ctx = km.load_context(cfg.context_file, cfg.task)
                except Exception as e:
                    logger.warning("[REPAIR] %s: could not load context (%s). Proceeding without.", p.name, e)

            # 컨텍스트 없으면 리페어 스킵(복사만)
            if method in {"rag", "kewrag"} and rag_ctx is None:
                # out_p = output_dir / p.name if CONFIG["REPAIR_KEEP_SAME_FILENAME"] else (output_dir / (p.stem + "__repair.csv"))
                try:
                    df_in.to_csv(out_p, index=False)
                except Exception as e:
                    logger.warning("[REPAIR] %s: skip_no_context but copy failed: %s", p.name, e)
                mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
                             missing_before, missing_before, 0, "skip_no_context"])
                mf.flush()
                continue

            try:
                df_out = km.repair_missing_predictions(
                    method=method,
                    llm=shared_llm,
                    cfg=cfg,
                    df=df_in,
                    max_retries=CONFIG["REPAIR_MAX_RETRIES"],
                    treat_unparseable_as_missing=CONFIG["REPAIR_TREAT_UNPARSEABLE_AS_MISSING"],
                    rag_context=rag_ctx,
                )
            except Exception as e:
                logger.exception("[REPAIR] %s: repair failed: %s", p.name, e)
                mw.writerow([tcga, printable, str(p), "", method, task, model_id, CONFIG["SEED"],
                             missing_before, "", "", f"repair_error: {e}"])
                mf.flush()
                continue

            # missing_after = count_missing_for_method(
            #     df_out[stage_col], task, treat_unparseable=CONFIG["REPAIR_TREAT_UNPARSEABLE_AS_MISSING"]
            # )
            missing_after = count_missing_for_method_in_df(
                df_out, method, task, treat_unparseable=CONFIG["REPAIR_TREAT_UNPARSEABLE_AS_MISSING"]
            )

            repaired = max(0, missing_before - missing_after)
            total_repaired += repaired

            # out_p = output_dir / p.name if CONFIG["REPAIR_KEEP_SAME_FILENAME"] else (output_dir / (p.stem + "__repair.csv"))
            try:
                df_out.to_csv(out_p, index=False)
            except Exception as e:
                logger.exception("[REPAIR] %s: failed to write output: %s", p.name, e)
                mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
                             missing_before, missing_after, repaired, f"write_error: {e}"])
                mf.flush()
                continue

            logger.info("[REPAIR] %s -> %s | repaired=%d (missing %d -> %d)",
                        p.name, out_p.name, repaired, missing_before, missing_after)
            mw.writerow([tcga, printable, str(p), str(out_p), method, task, model_id, CONFIG["SEED"],
                         missing_before, missing_after, repaired, "ok"])
            mf.flush()

    logger.info(f"\nRepair pass done. Manifest: {manifest_path}\n")
    return total_repaired

def run_repairs_iterative() -> None:
    """
    runs -> runs2 -> runs3 ... 식으로 반복 리페어 수행.
    라운드별로 입력/출력 디렉토리를 바꾸며, 수정된 행이 0이면(옵션) 조기 종료.
    """
    setup_logging(None, CONFIG["LOG_LEVEL"])
    root = Path(CONFIG["DATA_ROOT"]).resolve()

    rounds = int(CONFIG.get("REPAIR_ITERATE_ROUNDS", 1))
    stop_early = bool(CONFIG.get("REPAIR_STOP_EARLY", True))

    # 반복 1회일 때는 REPAIR_INPUT_DIR -> REPAIR_OUTPUT_DIR로
    if rounds <= 1:
        input_dir = (root / CONFIG["REPAIR_INPUT_DIR"]).resolve()
        output_dir = (root / CONFIG["REPAIR_OUTPUT_DIR"]).resolve()
        # setup_logging(None, CONFIG["LOG_LEVEL"])
        total = run_repairs(input_dir, output_dir, round_tag="r1")
        logger.info(f"[REPAIR] round=1, repaired_total={total}")
        return

    # 2회 이상일 때는 runs{index} 패턴 사용
    pattern = str(CONFIG.get("REPAIR_OUTPUT_DIR_PATTERN", "runs{}"))
    first_idx = int(CONFIG.get("REPAIR_FIRST_INDEX", 2))

    # 첫 입력은 REPAIR_INPUT_DIR
    cur_input = (root / CONFIG["REPAIR_INPUT_DIR"]).resolve()

    no_improve_rounds = 0 
    for r in range(rounds):
        round_idx = first_idx + r
        cur_output = (root / pattern.format(round_idx)).resolve()
        # setup_logging(None, CONFIG["LOG_LEVEL"])
        logger.info(f"[REPAIR] round={r+1} | {cur_input} -> {cur_output}")

        total = run_repairs(cur_input, cur_output, round_tag=f"r{r+1}")
        logger.info(f"[REPAIR] round={r+1}, repaired_total={total}")

        if stop_early: 
            if total == 0:
                no_improve_rounds += 1
            else:
                no_improve_rounds = 0
            patience = int(CONFIG.get("REPAIR_PATIENCE", 0))
            if no_improve_rounds > patience:
                logger.info(f"[REPAIR] No improvement for {no_improve_rounds} consecutive round(s). Early stopping.")
                break

        # 다음 라운드 입력은 방금 출력 디렉토리
        cur_input = cur_output

if __name__ == "__main__":
    setup_logging(None, CONFIG["LOG_LEVEL"])

    if CONFIG.get("REPAIR_ONLY", False):
        run_repairs_iterative()
    else:
        run_all()
        if CONFIG.get("REPAIR_AFTER_RUN", False):
            run_repairs_iterative()
