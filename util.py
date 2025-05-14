import json
import logging
logger = logging.getLogger(__name__)

def safe_json_load(s: str) -> any:
    """
    Attempts to parse a JSON string using multiple parsers.
    Order:
    1. json.loads (strict)
    2. demjson3.decode (tolerant)
    3. json5.loads (allows single quotes, unquoted keys, etc.)
    4. dirtyjson.loads (for messy JSON)
    5. jsom (if available)
    6. json_repair (attempt to repair the JSON and parse it)
    
    If all attempts fail, returns the original string.
    """
    # 1. Try standard JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.error("Standard json.loads failed: %s", e)
    
    # 2. Try demjson3
    try:
        import demjson3
        logger.info("Attempting to parse with demjson3 as fallback.")
        result = demjson3.decode(s)
        logger.info("demjson3 successfully parsed the JSON.")
        return result
    except Exception as e2:
        logger.error("demjson3 fallback failed: %s", e2)
    
    # 3. Try json5
    try:
        import json5
        logger.info("Attempting to parse with json5 as fallback.")
        result = json5.loads(s)
        logger.info("json5 successfully parsed the JSON.")
        return result
    except Exception as e3:
        logger.error("json5 fallback failed: %s", e3)
    
    # 4. Try dirtyjson
    try:
        import dirtyjson
        logger.info("Attempting to parse with dirtyjson as fallback.")
        result = dirtyjson.loads(s)
        logger.info("dirtyjson successfully parsed the JSON.")
        return result
    except Exception as e4:
        logger.error("dirtyjson fallback failed: %s", e4)
    
    # 5. Try jsom
    try:
        import jsom
        logger.info("Attempting to parse with jsom as fallback.")
        parser = jsom.JsomParser()
        result = parser.loads(s)
        logger.info("jsom successfully parsed the JSON.")
        return result
    except Exception as e5:
        logger.error("jsom fallback failed: %s", e5)
    
    # 6. Try json_repair (attempt to fix the JSON and then load it)
    try:
        import json_repair
        logger.info("Attempting to repair JSON with json_repair as fallback.")
        repaired = json_repair.repair_json(s)
        result = json.loads(repaired)
        logger.info("json_repair successfully parsed the JSON.")
        return result
    except Exception as e6:
        logger.error("json_repair fallback failed: %s", e6)
    
    # All attempts failed; return the original input.
    logger.error("All JSON parsing attempts failed. Returning None.")
    logger.error("Original input: %s", s)
    return None

if __name__ == "__main__":
    # Example usage
    json_string = '{"key": "value", "number": 123, "array": [1, 2, 3]'
    result = safe_json_load(json_string)
    print(result)  # Should print the parsed JSON object
    print(result.get("array"))  # Should print "value"