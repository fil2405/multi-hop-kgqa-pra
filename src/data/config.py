#type: ignore
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_RAW = BASE_DIR / "dataset_raw_KG"

PATH_KB = DATA_RAW / "kb.txt"

# split: "train", "dev", "test"

def get_qa_path(hop: int, split: str) -> Path:
    return DATA_RAW / f"{hop}-hop" / "vanilla" / f"qa_{split}.txt"

