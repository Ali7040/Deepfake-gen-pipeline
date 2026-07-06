"""Isolate the melt trigger: source-selection vs target-angle vs pixel-boost."""
import cv2
import numpy as np
import simple_app as app
from simple_app import (state_manager, face_analyser, face_swapper,
                        detect_faces_in_frame, _image_pixel_boost)

MAN  = 'uploads/source_1782302496_WhatsApp_Image_2026-06-24_at_4.55.18_PM.jpeg'   # 4 faces
TOM  = 'uploads/target_1782302496_WhatsApp_Image_2026-06-24_at_4.56.42_PM.jpeg'   # angled
FRONTAL = 'uploads/target_1780647643_angelina-jolie1-2d2e5526559a485588cbf8873af5f2af.jpg'

print('init:', app.initialize_deeptrace()); app.warmup_models(); app._apply_quality_profile('realistic')

def faces(path):
    return detect_faces_in_frame(cv2.imread(path))

def largest(fs):
    return max(fs, key=lambda f:(f.bounding_box[2]-f.bounding_box[0])*(f.bounding_box[3]-f.bounding_box[1]))

def leftmost(fs):
    return min(fs, key=lambda f: f.bounding_box[0])

man_faces = faces(MAN)
print(f'MAN source faces: {len(man_faces)}')
for i,f in enumerate(sorted(man_faces,key=lambda f:f.bounding_box[0])):
    bb=f.bounding_box; area=int((bb[2]-bb[0])*(bb[3]-bb[1]))
    print(f'  face{i}: x={bb[0]:.0f} area={area} detscore={f.score_set.get("detector"):.2f}')

src_large = largest(man_faces)
src_left  = leftmost(man_faces)

def run(tag, src_face, tgt_path, boost):
    state_manager.set_item('face_swapper_pixel_boost', _image_pixel_boost(boost))
    tgt = cv2.imread(tgt_path)
    tf = largest(faces(tgt_path))
    out = face_swapper.swap_face(src_face, tf, tgt.copy())
    p = f'outputs/_mtx_{tag}.jpg'
    cv2.imwrite(p, out)
    d = round(float(np.mean(np.abs(out.astype(int)-tgt.astype(int)))),2)
    print(f'[{tag}] boost={boost} diff_vs_tgt={d} -> {p}')

run('A_large_frontal_256', src_large, FRONTAL, '256x256')
run('B_large_tom_256',     src_large, TOM,     '256x256')
run('C_large_tom_512',     src_large, TOM,     '512x512')
run('D_left_tom_512',      src_left,  TOM,     '512x512')
print('DONE')
