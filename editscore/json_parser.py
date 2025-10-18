import json
import re
from typing import Dict, Any, List, Optional

# ==============================================================================
#  HELPER FUNCTIONS (Based on your provided robust fixers)
# ==============================================================================
# For clarity, these are named as internal functions (prefixed with an underscore).

def _fix_json_quotes(s: str) -> str:
    """First-stage repair: handle incorrect quotes and basic structure."""
    # Replace Python-style booleans/None with JSON standard
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    
    # Attempt to replace single quotes with double quotes (a common VLM error)
    # This is a high-risk operation that might break the reasoning content, 
    # but it's worth trying early on.
    try:
        temp_s = s.replace("'", '"')
        json.loads(temp_s)
        return temp_s
    except json.JSONDecodeError:
        # If it's still invalid after replacement, return the original string for the next repair step.
        pass

    # Add double quotes to keys (e.g., {reasoning: ...} -> {"reasoning": ...})
    s = re.sub(r'([\{\s,])(\w+)\s*:', r'\1"\2":', s)
    return s

def _repair_reasoning_field_robust(json_str: str) -> str:
    """Second-stage repair: specifically fix unescaped double quotes inside the 'reasoning' field."""
    pattern = re.compile(
        r'("reasoning"\s*:\s*")'  # --- Group 1: "reasoning" : "
        r'(.*?)'                  # --- Group 2: The content (non-greedy)
        r'(?="\s*[,}])',          # --- Lookahead: find " followed by , or }
        re.DOTALL
    )

    def replacer(match):
        prefix = match.group(1)
        content = match.group(2)
        # In the content, replace all unescaped " with \"
        fixed_content = content.replace('"', '\\"')
        return prefix + fixed_content

    return pattern.sub(replacer, json_str)

def _fallback_extract_and_rebuild(input_str: str) -> str:
    """Final fallback strategy: abandon repair, directly extract information, and rebuild a valid JSON."""
    # 1. Extract reasoning
    # Find all content between "reasoning": and ,"score":
    reasoning_text = ""
    reason_match = re.search(r'["\']reasoning["\']\s*:\s*["\']?(.*?)["\']?\s*,\s*["\']score["\']', input_str, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reasoning_text = reason_match.group(1).strip()
        # Clean up any potentially remaining escape characters
        reasoning_text = reasoning_text.replace('\\"', '"')
    else:
        # If not found, assume all text besides the score part is the reasoning.
        # First, remove the score part.
        score_part_match = re.search(r'["\']score["\']\s*:.*', input_str, re.IGNORECASE)
        if score_part_match:
            reasoning_text = input_str[:score_part_match.start()].strip()
        else:
            # If even 'score' cannot be found, assume the entire string is the reasoning.
            reasoning_text = input_str
            
    # 2. Extract scores
    scores = []
    # Prioritize searching after the 'score' keyword.
    score_match = re.search(r'["\']score["\']\s*:\s*(.*)', input_str, re.DOTALL | re.IGNORECASE)
    search_area = score_match.group(1) if score_match else input_str
    
    # Find all integers or floats.
    numbers = re.findall(r'[-+]?\d*\.?\d+', search_area)
    if numbers:
        scores = [float(num) for num in numbers]

    # 3. Rebuild into a standard dictionary and return a JSON string.
    rebuilt_data = {
        "reasoning": reasoning_text,
        "score": scores
    }
    return json.dumps(rebuilt_data, ensure_ascii=False)


def _format_and_validate_dict(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and format the parsed dictionary to ensure it meets the final output standard."""
    if not isinstance(data, dict):
        return None

    # Extract reasoning, tolerating case and spelling variations.
    reasoning = ""
    for key in ["reasoning", "reason", "rationale"]:
        if key in data and isinstance(data[key], str):
            reasoning = data[key]
            break

    # Extract score, and ensure it is a list of floats.
    scores = []
    if 'score' in data:
        score_val = data['score']
        if isinstance(score_val, list):
            scores = [float(s) for s in score_val if isinstance(s, (int, float, str))]
        elif isinstance(score_val, (int, float)):
            scores = [float(score_val)]
    
    # If any field was found, consider it a success.
    if reasoning or scores:
        return {"score": scores, "reasoning": reasoning}
    
    return None

# ==============================================================================
#  MAIN PARSING FUNCTION
# ==============================================================================

def parse_vlm_output_to_dict(input_string: str) -> Dict[str, Any]:
    """
    A highly robust function to parse a VLM's output string into a dictionary
    containing 'score' and 'reasoning'.

    It uses a multi-stage repair pipeline, progressively degrading from standard
    JSON parsing to a final information extraction fallback.
    """
    # --- 0. Preprocessing ---
    if not input_string or not input_string.strip():
        return {"score": [], "reasoning": "Input was empty."}
    
    # Find the substring enclosed by `{}`, which is often the core of the VLM output.
    json_match = re.search(r'\{.*\}', input_string, re.DOTALL)
    target_str = json_match.group(0) if json_match else input_string.strip()

    # --- Repair Pipeline ---
    # Apply fixers in order, attempting to parse after each one.
    
    fixer_pipeline = [
        lambda s: s,                        # 1. Try the original string.
        _fix_json_quotes,                   # 2. Fix basic quotes and keywords.
        _repair_reasoning_field_robust,     # 3. Fix internal quotes in the reasoning field.
    ]

    for fixer in fixer_pipeline:
        try:
            fixed_str = fixer(target_str)
            data = json.loads(fixed_str)
            validated_data = _format_and_validate_dict(data)
            if validated_data is not None:
                return validated_data
        except (json.JSONDecodeError, TypeError):
            # If it fails, continue to the next fixer.
            continue
            
    # --- Final Fallback Strategy ---
    # If all repair and parsing attempts fail, activate the information extraction mode.
    try:
        fallback_str = _fallback_extract_and_rebuild(target_str)
        # This function guarantees a valid JSON string, so we can load it directly.
        data = json.loads(fallback_str)
        # Still run it through the validator to standardize the format.
        validated_data = _format_and_validate_dict(data)
        if validated_data:
            return validated_data
    except Exception:
        # If even the final fallback strategy fails, return an error message.
        pass
        
    return {
        "score": [],
        "reasoning": f"Failed to parse after all strategies. Original output: '{input_string}'"
    }
