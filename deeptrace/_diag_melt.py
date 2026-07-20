"""Isolate WHERE the angled-face swap melts: dump each pipeline stage separately."""
import cv2
import numpy as np
import simple_app as app
from simple_app import (state_manager, face_analyser, face_swapper, face_enhancer,
                        detect_faces_in_frame, _advanced_face_blend, _image_pixel_boost)

SRC = 'uploads/source_1782302496_WhatsApp_Image_2026-06-24_at_4.55.18_PM.jpeg'
TGT = 'uploads/target_1782302496_WhatsApp_Image_2026-06-24_at_4.56.42_PM.jpeg'

print('init:', app.initialize_deeptrace())
app.warmup_models()
app._apply_quality_profile('realistic')
state_manager.set_item('face_swapper_pixel_boost', _image_pixel_boost('512x512'))

src = cv2.imread(SRC); tgt = cv2.imread(TGT)
print('src shape', src.shape, '| tgt shape', tgt.shape)

src_faces = detect_faces_in_frame(src)
tgt_faces = detect_faces_in_frame(tgt)
print(f'source faces: {len(src_faces)} | target faces: {len(tgt_faces)}')
if not src_faces or not tgt_faces:
    raise SystemExit('missing face(s)')

sf = src_faces[0]
tf = tgt_faces[0]
# Report target face geometry (angle proxies)
print('target bbox:', [round(x,1) for x in tf.bounding_box])
print('target angles attr:', getattr(tf, 'angle', None))
lm = tf.landmark_set.get('5/68')
print('target 5-landmarks:\n', np.round(lm, 1) if lm is not None else None)

# STAGE 1: raw swap only
state_manager.set_item('face_swapper_pixel_boost', _image_pixel_boost('512x512'))
s1 = face_swapper.swap_face(sf, tf, tgt.copy())
cv2.imwrite('outputs/_melt_1_rawswap.jpg', s1)
print('stage1 raw swap written; diff_vs_tgt=', round(float(np.mean(np.abs(s1.astype(int)-tgt.astype(int)))),2))

# STAGE 2: + enhancer
s2 = face_enhancer.enhance_face(tf, s1.copy())
cv2.imwrite('outputs/_melt_2_enhanced.jpg', s2)
print('stage2 enhanced written; diff_vs_stage1=', round(float(np.mean(np.abs(s2.astype(int)-s1.astype(int)))),2))

# STAGE 3: + advanced blend (LAB + unsharp + ellipse mask)
s3 = _advanced_face_blend(s2.copy(), tgt, [tf])
cv2.imwrite('outputs/_melt_3_full.jpg', s3)
print('stage3 full written; diff_vs_stage2=', round(float(np.mean(np.abs(s3.astype(int)-s2.astype(int)))),2))
print('DONE — inspect outputs/_melt_1_rawswap.jpg, _melt_2_enhanced.jpg, _melt_3_full.jpg')
