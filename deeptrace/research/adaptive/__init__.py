"""
Closed-loop quality-aware adaptive swapping (Contribution B).

Stock FaceFusion uses one fixed pixel-boost resolution for every face and never
checks the result. On CPU this forces a single quality/speed point for the whole
video. This package makes the swap quality-aware and budget-aware:

    Stage 0 (free)   pick a starting pixel-boost from the face's size in frame
                     (small faces gain nothing from high resolution).
    Stage 1 (cheap)  score the swap with ArcFace identity similarity to the
                     source; if it is below target AND not already at max
                     resolution, re-swap once at a higher resolution and keep
                     whichever scored higher.

Most frames stop at Stage 1, so extra compute is spent only on the hard frames
- the core "adaptive compute under a CPU budget" idea. Enabled by
DEEPTRACE_ADAPTIVE=1; disabled -> the configured resolution is used unchanged.
"""
