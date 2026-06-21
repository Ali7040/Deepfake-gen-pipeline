"""
Research feature flags.

All flags are read from environment variables and default to OFF. Using env
vars (instead of touching args.py / program.py right now) keeps this layer
fully decoupled from the stock CLI: with no env vars set, the pipeline behaves
exactly like upstream FaceFusion.

Environment variables
---------------------
    DEEPTRACE_INSTRUMENT   "1" to record per-swap quality/timing metrics  (read-only)
    DEEPTRACE_METRICS_PATH path to the JSONL metrics file
                           (default: outputs/deeptrace_metrics.jsonl)
    DEEPTRACE_TEMPORAL     "1" to enable landmark stabilization             (Step 2)
    DEEPTRACE_FLOW_BLEND   "1" to enable motion-compensated output blending  (Step 4)
    DEEPTRACE_ADAPTIVE     "1" to enable the closed-loop quality controller  (Step 5)

Temporal and flow-blend are independent toggles so the ablation study can
attribute the effect of each. Either one forces single-threaded video
processing (both need frames in order).

Later steps may promote these to first-class CLI flags; the defaults here must
stay OFF regardless.
"""

import os
from dataclasses import dataclass


def _flag(name : str) -> bool:
	return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass
class ResearchConfig:
	instrument : bool
	metrics_path : str
	temporal : bool
	flow_blend : bool
	adaptive : bool

	@classmethod
	def from_env(cls) -> 'ResearchConfig':
		return cls(
			instrument = _flag('DEEPTRACE_INSTRUMENT'),
			metrics_path = os.environ.get('DEEPTRACE_METRICS_PATH', 'outputs/deeptrace_metrics.jsonl'),
			temporal = _flag('DEEPTRACE_TEMPORAL'),
			flow_blend = _flag('DEEPTRACE_FLOW_BLEND'),
			adaptive = _flag('DEEPTRACE_ADAPTIVE'),
		)

	@property
	def needs_sequential(self) -> bool:
		# Any layer that filters across frames needs in-order processing.
		return self.temporal or self.flow_blend

	@property
	def any_enabled(self) -> bool:
		return self.instrument or self.temporal or self.flow_blend or self.adaptive


# Evaluated once at import. Set env vars before launching the process.
research_config = ResearchConfig.from_env()
