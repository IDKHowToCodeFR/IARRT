"""Download and cache a small OPUS German-English subset for IARRT."""

import json
import os
import random
from typing import Any, Dict

import pandas as pd
from datasets import load_dataset


DATA_DIR = os.path.abspath(os.path.dirname(__file__))
SUBSET_CSV = os.path.join(DATA_DIR, "subset.csv")
IDIOM_JSON = os.path.join(DATA_DIR, "idiom_dictionary.json")

# Expanded idiom list for the upgraded prototype
EXPANDED_IDIOMS: Dict[str, str] = {
    "ins kalte wasser springen": "take the plunge",
    "die katze aus dem sack lassen": "let the cat out of the bag",
    "den nagel auf den kopf treffen": "hit the nail on the head",
    "um den heissen brei reden": "beat around the bush",
    "tomaten auf den augen haben": "fail to see what is obvious",
    "nur bahnhof verstehen": "not understand a thing",
    "daumen druecken": "keep one's fingers crossed",
    "das kind beim namen nennen": "call a spade a spade",
    "zwei fliegen mit einer klappe schlagen": "kill two birds with one stone",
    "jemandem einen baeren aufbinden": "pull someone's leg",
    "das ruder herumreissen": "turn things around",
    "die flinte ins korn werfen": "throw in the towel",
    "die kirche im dorf lassen": "not get carried away",
    "auf dem holzweg sein": "be on the wrong track",
    "den faden verlieren": "lose the thread",
    "ein auge zudruecken": "turn a blind eye",
    "da liegt der hund begraben": "that is the heart of the matter",
    "jemanden auf den arm nehmen": "pull someone's leg",
    "nichts anbrennen lassen": "stay on top of things",
    "unter vier augen": "in private",
    "einen kühlen kopf bewahren": "keep a cool head",
    "ins fettnäpfchen treten": "put one's foot in it",
    "die nase vorn haben": "be ahead of the game",
    "schwein haben": "be lucky",
    "ein alter hase": "an old hand",
    "auf achse sein": "be on the move",
    "blau sein": "be drunk",
    "fix und fertig sein": "be exhausted",
    "einen vogel haben": "be crazy",
    "die nase voll haben": "be fed up",
}


def save_idiom_dictionary():
    """Save the expanded idiom dictionary to a JSON file."""
    with open(IDIOM_JSON, "w") as f:
        json.dump(EXPANDED_IDIOMS, f, indent=4)
    print(f"Saved expanded dictionary to {IDIOM_JSON}")


def _extract_columns(dataset: Any) -> pd.DataFrame:
    """Handle common HuggingFace parallel-data schemas."""
    names = dataset.column_names

    if "de" in names and "en" in names:
        return pd.DataFrame({"de": dataset["de"], "en": dataset["en"]})

    if "translation" in names:
        german = []
        english = []
        for item in dataset["translation"]:
            german.append(item.get("de") or item.get("ger") or "")
            english.append(item.get("en") or item.get("eng") or "")
        return pd.DataFrame({"de": german, "en": english})

    if "text" in names:
        german = []
        english = []
        for item in dataset["text"]:
            if "###>" not in item:
                continue
            de_text, en_text = item.split("###>", maxsplit=1)
            german.append(de_text.strip())
            english.append(en_text.strip())
        return pd.DataFrame({"de": german, "en": english})

    raise ValueError(f"Could not find German/English columns in dataset: {names}")


def download_and_save_subset(sample_size: int = 4000, seed: int = 42) -> str:
    """Download the dataset, sample examples, and write data/subset.csv."""
    random.seed(seed)
    save_idiom_dictionary()

    if os.path.exists(SUBSET_CSV):
        cached = pd.read_csv(SUBSET_CSV)
        if len(cached) >= min(sample_size, 2000):
            print(f"Using cached subset at {SUBSET_CSV}")
            return SUBSET_CSV

    ds = load_dataset("kaitchup/opus-German-to-English", split="train")
    ds = ds.shuffle(seed=seed)
    subset = ds.select(range(min(sample_size, len(ds))))
    df = _extract_columns(subset).dropna()

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(SUBSET_CSV, index=False)
    print(f"Saved subset of {len(df)} pairs to {SUBSET_CSV}")
    return SUBSET_CSV


if __name__ == "__main__":
    download_and_save_subset()
