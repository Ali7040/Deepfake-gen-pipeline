"""
Compare a baseline run against a temporal (or any improved) run.

Examples
--------
Flicker only (videos):
    python -m deeptrace.research.eval.compare \
        --baseline-video outputs/out_baseline.mp4 \
        --temporal-video outputs/out_temporal.mp4

Identity stability (metrics JSONL from instrumentation):
    python -m deeptrace.research.eval.compare \
        --baseline-metrics outputs/metrics_baseline.jsonl \
        --temporal-metrics outputs/metrics_temporal.jsonl

You can pass both pairs at once to produce the full ablation row.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

from deeptrace.research.eval.temporal_metrics import identity_consistency, warping_error


def _load_jsonl(path : str) -> List[Dict[str, Any]]:
	rows = []
	with open(path, 'r', encoding = 'utf-8') as handle:
		for line in handle:
			line = line.strip()
			if line:
				rows.append(json.loads(line))
	return rows


def _delta(base : Optional[float], new : Optional[float], lower_is_better : bool) -> str:
	if base is None or new is None:
		return 'n/a'
	change = new - base
	if abs(base) > 1e-9:
		pct = (change / abs(base)) * 100.0
	else:
		pct = 0.0
	improved = (change < 0) if lower_is_better else (change > 0)
	arrow = 'better' if improved else 'worse'
	return '{:+.4f} ({:+.1f}%, {})'.format(change, pct, arrow)


def _row(label : str, base : Optional[float], new : Optional[float], lower_is_better : bool) -> None:
	base_s = 'n/a' if base is None else '{:.4f}'.format(base)
	new_s = 'n/a' if new is None else '{:.4f}'.format(new)
	print('  {:<22} baseline={:<10} improved={:<10} {}'.format(label, base_s, new_s, _delta(base, new, lower_is_better)))


def main() -> None:
	parser = argparse.ArgumentParser(description = 'Temporal-consistency comparison')
	parser.add_argument('--baseline-video')
	parser.add_argument('--temporal-video')
	parser.add_argument('--baseline-metrics')
	parser.add_argument('--temporal-metrics')
	parser.add_argument('--max-frames', type = int, default = None)
	args = parser.parse_args()

	print('=' * 72)
	print('DeepTrace temporal-consistency comparison')
	print('=' * 72)

	if args.baseline_video and args.temporal_video:
		print('\n[Flicker]  warping error (lower = less flicker)')
		base = warping_error(args.baseline_video, args.max_frames)
		new = warping_error(args.temporal_video, args.max_frames)
		_row('warping_error', base.get('warping_error'), new.get('warping_error'), lower_is_better = True)

	if args.baseline_metrics and args.temporal_metrics:
		base_rows = _load_jsonl(args.baseline_metrics)
		new_rows = _load_jsonl(args.temporal_metrics)
		base_id = identity_consistency(base_rows)
		new_id = identity_consistency(new_rows)
		print('\n[Identity stability]  (higher = more stable)')
		_row('TL-ID (consecutive)', base_id.get('tl_id'), new_id.get('tl_id'), lower_is_better = False)
		_row('TG-ID (vs clip mean)', base_id.get('tg_id'), new_id.get('tg_id'), lower_is_better = False)

		def _mean(rows, key):
			vals = [r.get(key) for r in rows if r.get(key) is not None]
			return sum(vals) / len(vals) if vals else None

		print('\n[Quality]  (higher ID-Sim = closer to source; higher sharpness = crisper)')
		_row('ID-Sim (vs source)', _mean(base_rows, 'id_sim'), _mean(new_rows, 'id_sim'), lower_is_better = False)
		_row('sharpness', _mean(base_rows, 'sharpness'), _mean(new_rows, 'sharpness'), lower_is_better = False)
		print('\n[Cost]')
		_row('elapsed_ms / swap', _mean(base_rows, 'elapsed_ms'), _mean(new_rows, 'elapsed_ms'), lower_is_better = True)

	print()


if __name__ == '__main__':
	main()
