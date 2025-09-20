import json
from typing import List, Dict
from pathlib import Path

def save_processed_data(data: List[Dict], filepath: str):
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_processed_data(filepath: str) -> List[Dict]:
    p = Path(filepath)
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_data_summary_report(data: List[Dict]) -> Dict:
    total = len(data)
    labeled = [d for d in data if d.get('label') is not None]
    pos = sum(1 for d in labeled if d.get('label')==1)
    neg = sum(1 for d in labeled if d.get('label')==0)
    authors = set()
    for d in data:
        if d.get('author1'): authors.add(d.get('author1'))
        if d.get('author2'): authors.add(d.get('author2'))
    return {
        'total_pairs': total,
        'labeled_pairs': len(labeled),
        'positive_pairs': pos,
        'negative_pairs': neg,
        'unique_authors': len(authors)
    }
