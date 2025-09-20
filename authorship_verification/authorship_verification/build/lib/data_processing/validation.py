from typing import List, Dict
import chardet

class DataValidator:
    def __init__(self, min_length: int = 30):
        self.min_length = min_length

    def validate_dataset(self, data: List[Dict]) -> Dict:
        issues = {
            'empty_texts': [],
            'very_short_texts': [],
            'duplicate_pairs': [],
            'label_imbalance': None,
            'encoding_issues': []
        }
        seen = set()
        label_counts = {}
        for idx, item in enumerate(data):
            t1 = item.get('text1','') or ''
            t2 = item.get('text2','') or ''
            if len(t1.strip())==0 or len(t2.strip())==0:
                issues['empty_texts'].append(idx)
            if len(t1) < self.min_length or len(t2) < self.min_length:
                issues['very_short_texts'].append(idx)
            key = (t1[:50], t2[:50])
            if key in seen:
                issues['duplicate_pairs'].append(idx)
            seen.add(key)
            lbl = item.get('label')
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            # encoding detection - lightweight
            try:
                _ = t1.encode('utf-8')
                _ = t2.encode('utf-8')
            except Exception:
                issues['encoding_issues'].append(idx)
        # label imbalance check
        if label_counts:
            total = sum(v for k,v in label_counts.items() if k is not None)
            pos = label_counts.get(1,0)
            neg = label_counts.get(0,0)
            imbalance = None
            if total > 0:
                imbalance = {'pos': pos, 'neg': neg, 'ratio_pos_neg': (pos/neg) if neg>0 else None}
            issues['label_imbalance'] = imbalance
        return issues

    def suggest_fixes(self, issues: Dict) -> List[str]:
        suggestions = []
        if issues.get('empty_texts'):
            suggestions.append('Remove or fill pairs with empty text; check data ingestion pipeline.')
        if issues.get('very_short_texts'):
            suggestions.append('Review very short texts: consider removing or merging with other samples.')
        if issues.get('duplicate_pairs'):
            suggestions.append('Remove duplicate pairs or deduplicate by hashing text content.')
        if issues.get('label_imbalance') and issues['label_imbalance'].get('ratio_pos_neg') is not None:
            ratio = issues['label_imbalance']['ratio_pos_neg']
            if ratio is not None and (ratio < 0.2 or ratio > 5):
                suggestions.append('Label imbalance detected: consider resampling or class weighting.')
        if issues.get('encoding_issues'):
            suggestions.append('Investigate encoding issues; try reading files with detected encodings.')
        return suggestions
