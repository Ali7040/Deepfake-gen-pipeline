"""Smoke test: compare realistic vs faithful swap on a cross-gender pair."""
import time
import numpy as np
import cv2
import simple_app as app

SRC = 'uploads/source_1780647643_269411_v9_bd.jpg'              # male
TGT = 'uploads/target_1780647643_angelina-jolie1-2d2e5526559a485588cbf8873af5f2af.jpg'  # female

print('init:', app.initialize_deeptrace())
app.warmup_models()
print('ENHANCER_ACTIVE =', app._ENHANCER_ACTIVE)

tgt = cv2.imread(TGT)

def run(label, gmm, out):
    app._apply_quality_profile('faithful' if gmm == 'faithful' else 'realistic')
    t0 = time.time()
    res = app.process_image_swap(SRC, TGT, out, gender_match_mode=gmm)
    res = {k: v for k, v in res.items() if k != 'preview_b64'}
    o = cv2.imread(out)
    diff = float(np.mean(np.abs(o.astype(int) - tgt.astype(int)))) if (o is not None and o.shape == tgt.shape) else -1
    print(f'[{label}] {res} | elapsed={round(time.time()-t0,1)}s | mean_abs_diff_vs_target={round(diff,2)}')

run('realistic', 'features_only', 'outputs/_smoke_realistic.jpg')
run('faithful',  'faithful',      'outputs/_smoke_faithful.jpg')

# Negative test: a target with no face should now FAIL loudly, not return the input.
import os
blank = 'outputs/_blank.png'
cv2.imwrite(blank, np.full((400, 400, 3), 127, np.uint8))
res = app.process_image_swap(SRC, blank, 'outputs/_smoke_noface.png', gender_match_mode='features_only')
print('[no-face target]', {k: v for k, v in res.items() if k != 'preview_b64'})
print('DONE')
