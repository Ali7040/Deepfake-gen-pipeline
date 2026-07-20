"""
Summarise a metrics JSONL file produced by the instrumentation layer.

Usage:
    python -m deeptrace.research.report [path/to/metrics.jsonl]

Prints mean / median ID-Sim, sharpness and seconds-per-frame, grouped by the
swapper model that produced each row. This is what you paste into the baseline
row of the ablation table.
"""

import json
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List

from deeptrace.research.config import research_config


def load_rows(path : str) -> List[Dict[str, Any]]:
	rows = []
	with open(path, 'r', encoding = 'utf-8') as metrics_file:
		for line in metrics_file:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def _summarise(values : List[float]) -> str:
	values = [value for value in values if value is not None]
	if not values:
		return 'n/a'
	mean = statistics.mean(values)
	median = statistics.median(values)
	return 'mean={:.4f}  median={:.4f}  n={}'.format(mean, median, len(values))


def report(path : str) -> None:
	rows = load_rows(path)
	if not rows:
		print('No rows found in ' + path)
		return

	groups : Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	for row in rows:
		groups[row.get('model_name') or 'unknown'].append(row)

	print('=' * 64)
	print('DeepTrace metrics report :  ' + path)
	print('total swaps recorded     :  {}'.format(len(rows)))
	print('=' * 64)

	for model_name, model_rows in sorted(groups.items()):
		print('\nmodel: {}   (pixel_boost={})'.format(model_name, model_rows[0].get('pixel_boost')))
		print('  ID-Sim     : ' + _summarise([row.get('id_sim') for row in model_rows]))
		print('  sharpness  : ' + _summarise([row.get('sharpness') for row in model_rows]))
		print('  elapsed_ms : ' + _summarise([row.get('elapsed_ms') for row in model_rows]))

	all_ms = [row.get('elapsed_ms') for row in rows if row.get('elapsed_ms') is not None]
	if all_ms:
		print('\noverall seconds/swap : {:.3f}'.format(statistics.mean(all_ms) / 1000.0))


if __name__ == '__main__':
	target_path = sys.argv[1] if len(sys.argv) > 1 else research_config.metrics_path
	report(target_path)
