"""Diagnostic: run CURRENT pipeline on the SAME real-photo pair that produced
the good _AFTER_realistic.jpg, so we can tell a pipeline regression apart from
the synthetic-input look.
"""
import time
import cv2
import numpy as np
import simple_app as app

SRC = 'uploads/source_1780647643_269411_v9_bd.jpg'
TGT = 'uploads/target_1780647643_angelina-jolie1-2d2e5526559a485588cbf8873af5f2af.jpg'
REF = 'outputs/_AFTER_realistic.jpg'   # the known-good output

print('init:', app.initialize_deeptrace())
app.warmup_models()
print('ENHANCER_ACTIVE =', app._ENHANCER_ACTIVE)
app._apply_quality_profile('realistic')

out = 'outputs/_diag_current_realistic.jpg'
t0 = time.time()
res = app.process_image_swap(SRC, TGT, out, gender_match_mode='features_only')
res = {k: v for k, v in res.items() if k != 'preview_b64'}
print('RESULT:', res, '| elapsed', round(time.time() - t0, 1), 's')

cur = cv2.imread(out)
ref = cv2.imread(REF)
if cur is not None and ref is not None and cur.shape == ref.shape:
    diff = float(np.mean(np.abs(cur.astype(int) - ref.astype(int))))
    print(f'mean_abs_diff(current_vs_known_good) = {diff:.2f}  (0 = identical to good ref)')
else:
    print(f'shape mismatch: current={None if cur is None else cur.shape} ref={None if ref is None else ref.shape}')
print('wrote', out)
