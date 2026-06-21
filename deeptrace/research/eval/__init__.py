"""
Temporal-consistency evaluation (Step 3).

Turns the qualitative claim "the temporal layer reduces flicker" into hard
numbers for the paper:

    temporal_metrics.warping_error          - optical-flow flicker metric on a
                                               single output video (standalone,
                                               needs only OpenCV + the video).
    temporal_metrics.identity_consistency   - TL-ID / TG-ID from the per-frame
                                               embeddings logged by the
                                               instrumentation layer.

    compare (CLI)                           - run the above on a baseline vs a
                                               temporal run and print the delta.
"""
