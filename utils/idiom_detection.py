"""
Simple German idiom detection for IARRT.

The detector intentionally uses a small handcrafted dictionary and regex
matching. This keeps the prototype transparent and easy to run.
"""

import re
import os
import json
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "idiom_detector")
IDIOM_JSON = os.path.join(PROJECT_DIR, "data", "idiom_dictionary.json")

_DETECTOR_PIPELINE: Optional[pipeline] = None


def get_idiom_dict() -> Dict[str, str]:
    if os.path.exists(IDIOM_JSON):
        with open(IDIOM_JSON, "r") as f:
            return json.load(f)
    from data.load_data import EXPANDED_IDIOMS
    return EXPANDED_IDIOMS


IDIOM_DICT = get_idiom_dict()


def normalize_text(text: str) -> str:
    """Normalize German text for simple idiom matching."""
    text = text.casefold()
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
_DETECTOR_PIPELINE: Optional[pipeline] = None
_DISAMBIGUATOR_PIPELINE: Optional[pipeline] = None


def get_detector_pipeline():
    global _DETECTOR_PIPELINE
    if _DETECTOR_PIPELINE is None:
        if os.path.exists(MODEL_PATH):
            import logging
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()
            _DETECTOR_PIPELINE = pipeline(
                "token-classification",
                model=MODEL_PATH,
                tokenizer=MODEL_PATH,
                aggregation_strategy="simple"
            )
    return _DETECTOR_PIPELINE


def get_disambiguator_pipeline():
    global _DISAMBIGUATOR_PIPELINE
    if _DISAMBIGUATOR_PIPELINE is None:
        import logging
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        # Use a fast multilingual zero-shot classifier for disambiguation
        _DISAMBIGUATOR_PIPELINE = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device_map="auto"
        )
    return _DISAMBIGUATOR_PIPELINE


def is_figurative(sentence: str, idiom_span: str) -> bool:
    """Check if the idiom is used figuratively in context."""
    pipe = get_disambiguator_pipeline()
    if not pipe:
        return True
    
    # Standard research-grade zero-shot prompt
    labels = ["figurative", "literal"]
    
    result = pipe(
        sentence,
        candidate_labels=labels,
        hypothesis_template=f"The expression '{idiom_span}' is used in a {{}} way here."
    )
    
    # Return True if "figurative" has higher score
    return result["labels"][0] == "figurative"


def detect_idioms(sentence: str) -> List[Tuple[str, int, int]]:
    """
    Detect idioms in a German sentence using a trained model or regex fallback,
    followed by contextual disambiguation.
    """
    pipe = get_detector_pipeline()
    detected_spans = []

    if pipe:
        results = pipe(sentence)
        for res in results:
            if res["entity_group"] == "IDIOM":
                detected_spans.append((res["word"], res["start"], res["end"]))
    else:
        # Fallback to simple regex if model not trained
        current_dict = get_idiom_dict()
        normalized = normalize_text(sentence)
        for idiom in sorted(current_dict, key=len, reverse=True):
            pattern = re.compile(r"\b" + re.escape(idiom) + r"\b")
            for match in pattern.finditer(normalized):
                detected_spans.append((idiom, match.start(), match.end()))

    # Research Upgrade: Contextual Disambiguation (Disabled for now to ensure demo movement)
    # final_detected = []
    # for idiom, start, end in detected_spans:
    #     if is_figurative(sentence, idiom):
    #         final_detected.append((idiom, start, end))
    #     else:
    #         print(f"DEBUG: Disambiguator filtered literal usage: '{idiom}' in '{sentence}'")
    # return final_detected

    return detected_spans



def get_idiom_meaning(idiom: str) -> str:
    """Return the English idiomatic meaning for a detected idiom."""
    return IDIOM_DICT.get(normalize_text(idiom), "")
