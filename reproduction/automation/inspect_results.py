"""Inspect persisted predictions without importing torch or loading a GPU model."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--examples', type=int, default=3)
    a = p.parse_args()
    rows = []
    for number, line in enumerate(a.results.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise SystemExit(f'第 {number} 行 JSON 不完整；请备份后核查，不自动删除。')
    counts = Counter((r['key'], r['condition']) for r in rows)
    duplicates = [key for key, count in counts.items() if count > 1]
    print('总行数:', len(rows), '重复键:', duplicates)
    for condition in ('base', 'copollm'):
        current = [r for r in rows if r['condition'] == condition]
        print('\n条件:', condition, '行数:', len(current),
              '唯一样本:', len({r['key'] for r in current}),
              '高危行数:', sum(r['gold_risk'] == '高危' for r in current),
              '达到输出上限:', sum(r['hit_generation_limit'] for r in current))
        for row in current[:a.examples]:
            print(row['key'], row['gold_tags'], row['gold_risk'], repr(row['text']))
    if duplicates:
        raise SystemExit('存在重复键，不应直接据此重算指标。')

if __name__ == '__main__':
    main()
