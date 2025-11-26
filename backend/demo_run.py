from pathlib import Path
from src.data_processing.data_loader import AuthorshipDataLoader
from src.data_processing.preprocessing import TextPreprocessor
from src.data_processing.validation import DataValidator
from src.utils.data_utils import create_data_summary_report

def main():
    data_root = Path("data")
    print("Looking for data under:", data_root.resolve())

    loader = AuthorshipDataLoader(data_path=str(data_root), format_type="jsonl")
    splits = loader.load_data()  # {"train": [...], "validation": [...]}

    train = splits.get("train", [])
    val = splits.get("validation", [])
    print(f"Loaded: train={len(train)} records, validation={len(val)} records")

    if not train:
        print("No data found — ensure data/train/train.jsonl exists.")
        return

    pre = TextPreprocessor()
    for rec in train[:5]:
        txt = rec.get("text", "")
        rec["text_clean"] = pre.preprocess(txt)

    validator = DataValidator()
    issues = validator.validate_dataset(
        data=[{"text1": r.get("text", r.get("text_clean", "")), "text2": ""} for r in train]
    )
    print("Validation keys:", list(issues.keys()))

    report = create_data_summary_report(train)
    print("Data summary:", report)

if __name__ == "__main__":
    main()
