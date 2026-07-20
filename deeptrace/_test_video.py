"""Video face-swap test: swap one downloaded face onto the trimmed test clip."""
import os
import time

import cv2

import simple_app as app

SRC = 'uploads/online_test/face_01.jpg'
TGT = 'uploads/online_test/_vid_src.mp4'
OUT = 'outputs/online_test/_video_swap.mp4'

print('init:', app.initialize_deeptrace())
app.warmup_models()
app._apply_quality_profile('realistic')

t0 = time.time()
res = app.process_image_swap(SRC, TGT, OUT, gender_match_mode='features_only',
                             detect_interval=5, max_side=640)
res = {k: v for k, v in res.items() if k != 'preview_b64'}
elapsed = round(time.time() - t0, 1)
print('RESULT:', res, '| elapsed', elapsed, 's')

if os.path.exists(OUT):
    c = cv2.VideoCapture(OUT)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'OUTPUT video: {n} frames, {int(c.get(3))}x{int(c.get(4))}, exists OK')
    # Dump frame 0, 25, 49 as stills so we can eyeball motion/consistency
    for fi in (0, 25, 49):
        c.set(cv2.CAP_PROP_POS_FRAMES, fi)
        r, f = c.read()
        if r:
            cv2.imwrite(f'outputs/online_test/_vid_frame_{fi:02d}.jpg', f)
            print(f'  saved still frame {fi}')
    c.release()
else:
    print('NO OUTPUT WRITTEN')
