"""
Temporal-consistency layer (Contribution A).

Stock FaceFusion swaps every frame independently, so detector/landmarker jitter
turns into visible "swimming" and flicker in the swapped video. This package
removes that jitter WITHOUT any neural optical-flow network (none would run at a
usable speed on CPU):

    one_euro    - the 1-Euro filter (Casiez et al., 2012): a jitter-vs-lag
                  adaptive low-pass filter, O(1) per value.
    stabilizer  - applies 1-Euro to the 5-point face landmarks across frames,
                  per face track, in frame-index order.

Activated by DEEPTRACE_TEMPORAL=1. When disabled, callers bypass it entirely so
the stock landmarks (and therefore the stock output) are reproduced exactly.

Ordering requirement:
    1-Euro is a causal filter, so frames must be filtered in increasing
    frame-index order. The video workflow forces single-threaded execution while
    temporal mode is on (see workflows/image_to_video.py) to guarantee this.
"""
