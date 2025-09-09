from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

LOGGER_NAME = "kew_methods"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def setup_logging(log_file: Optional[str], level: str = "INFO", name: str = LOGGER_NAME) -> logging.Logger:
    """
    Safe to call multiple times.
    """
    logger = logging.getLogger(name)
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)
    logger.propagate = False

    # Remove existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.info("Logging initialized. Level=%s, File=%s", level.upper(), log_file or "(none)")
    return logger


def safe_json_load(s: str, logger: Optional[logging.Logger] = None) -> Any:
    """
    Attempts to parse a JSON string using multiple parsers.
    Order:
      1. json.loads (strict)
      2. demjson3.decode (tolerant)
      3. json5.loads (allows single quotes, unquoted keys, etc.)
      4. dirtyjson.loads (for messy JSON)
      5. jsom (if available)
      6. json_repair (attempt to repair the JSON and parse it)

    If all attempts fail, returns None.
    """
    log = logger or get_logger()
    # 1) Standard JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        log.error("Standard json.loads failed: %s", e)

    # 2) demjson3
    try:
        import demjson3  # type: ignore
        log.info("Attempting to parse with demjson3 as fallback.")
        result = demjson3.decode(s)
        log.info("demjson3 successfully parsed the JSON.")
        return result
    except Exception as e2:
        log.error("demjson3 fallback failed: %s", e2)

    # 3) json5
    try:
        import json5  # type: ignore
        log.info("Attempting to parse with json5 as fallback.")
        result = json5.loads(s)
        log.info("json5 successfully parsed the JSON.")
        return result
    except Exception as e3:
        log.error("json5 fallback failed: %s", e3)

    # 4) dirtyjson
    try:
        import dirtyjson  # type: ignore
        log.info("Attempting to parse with dirtyjson as fallback.")
        result = dirtyjson.loads(s)
        log.info("dirtyjson successfully parsed the JSON.")
        return result
    except Exception as e4:
        log.error("dirtyjson fallback failed: %s", e4)

    # 5) jsom
    try:
        import jsom  # type: ignore
        log.info("Attempting to parse with jsom as fallback.")
        parser = jsom.JsomParser()
        result = parser.loads(s)
        log.info("jsom successfully parsed the JSON.")
        return result
    except Exception as e5:
        log.error("jsom fallback failed: %s", e5)

    # 6) json_repair
    try:
        import json_repair  # type: ignore
        log.info("Attempting to repair JSON with json_repair as fallback.")
        repaired = json_repair.repair_json(s)
        result = json.loads(repaired)
        log.info("json_repair successfully parsed the JSON.")
        return result
    except Exception as e6:
        log.error("json_repair fallback failed: %s", e6)

    log.error("All JSON parsing attempts failed. Returning None.")
    # Do NOT log the entire raw content here to avoid duplicating; caller can log or store it.
    return None
