"""Batch test: run image->image face swaps over the 20 downloaded online faces.

Pairs faces (01->02, 03->04, ...) giving 10 swaps. For each pair we report
success, faces swapped, elapsed time, and mean abs pixel diff vs the target
(a non-trivial diff confirms the face region was actually regenerated, not a
no-op passthrough).
"""
import os
import glob
import time

import cv2
import numpy as np

import simple_app as app

FACES = sorted(glob.glob('uploads/online_test/face_*.jpg'))
OUT_DIR = 'outputs/online_test'
os.makedirs(OUT_DIR, exist_ok=True)

print('init:', app.initialize_deeptrace())
app.warmup_models()
print('ENHANCER_ACTIVE =', app._ENHANCER_ACTIVE)
app._apply_quality_profile('realistic')

pairs = [(FACES[i], FACES[i + 1]) for i in range(0, len(FACES) - 1, 2)]
results = []

for idx, (src, tgt) in enumerate(pairs, 1):
    out = os.path.join(OUT_DIR, f'swap_{idx:02d}.jpg')
    t0 = time.time()
    try:
        res = app.process_image_swap(src, tgt, out, gender_match_mode='features_only')
    except Exception as exc:  # noqa: BLE001
        res = {'success': False, 'error': f'EXCEPTION: {exc}'}
    elapsed = round(time.time() - t0, 1)

    tgt_img = cv2.imread(tgt)
    out_img = cv2.imread(out) if os.path.exists(out) else None
    if out_img is not None and tgt_img is not None and out_img.shape == tgt_img.shape:
        diff = round(float(np.mean(np.abs(out_img.astype(int) - tgt_img.astype(int)))), 2)
    else:
        diff = -1

    ok = res.get('success')
    n = res.get('faces_swapped', 0)
    err = '' if ok else (' | ' + str(res.get('error', '')))
    print(f'[{idx:02d}] {os.path.basename(src)} -> {os.path.basename(tgt)} '
          f'| ok={ok} faces={n} | {elapsed}s | diff_vs_target={diff}{err}')
    results.append((idx, ok, n, elapsed, diff))

ok_count = sum(1 for r in results if r[1])
avg_diff = np.mean([r[4] for r in results if r[4] >= 0]) if results else 0
avg_time = np.mean([r[3] for r in results]) if results else 0
print('\n==== IMAGE BATCH SUMMARY ====')
print(f'pairs run        : {len(results)}')
print(f'successful swaps : {ok_count}/{len(results)}')
print(f'avg elapsed/img  : {avg_time:.1f}s')
print(f'avg diff_vs_tgt  : {avg_diff:.1f} (higher = more of the face changed)')
print(f'outputs in       : {OUT_DIR}/')
