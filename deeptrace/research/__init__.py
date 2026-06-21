"""
DeepTrace research extensions.

This package contains all additive, opt-in research modules layered on top of
the stock face-swapping pipeline:

    config       - feature flags (all default OFF; baseline path is unchanged)
    metrics      - read-only quality measurements (ArcFace ID similarity, sharpness)
    instrument   - per-frame metric recorder + timing

Design contract:
    Nothing in this package may alter pipeline OUTPUT when its flag is disabled.
    When a flag is disabled the corresponding code path is bypassed entirely, so
    the stock FaceFusion result is reproduced byte-for-byte. This is required so
    that the stock pipeline can serve as the experimental control / baseline.
"""

from deeptrace.research.config import research_config
