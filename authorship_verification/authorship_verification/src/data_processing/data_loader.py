import json
from pathlib import Path
from typing import Dict, List, Any

class AuthorshipDataLoader:
    def __init__(self, data_path: str, format_type: str = "auto"):
        self.root = Path(data_path)
        self.format_type = format_type  # "auto" | "jsonl"

    def _read_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        train_p = self.root / "train" / "train.jsonl"
        val_p = self.root / "validation" / "val.jsonl"

        if self.format_type in ("auto", "jsonl"):
            if train_p.exists():
                train = self._read_jsonl(train_p)
                val = self._read_jsonl(val_p) if val_p.exists() else []
                return {"train": train, "validation": val}
            if self.format_type == "jsonl":
                raise FileNotFoundError(f"Expected JSONL at {train_p} but not found")

        raise ValueError(f"Unsupported or unknown format: {self.format_type}")
