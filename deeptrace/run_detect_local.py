#!/usr/bin/env python3
"""Run face detection from inside the deeptrace folder to avoid import collisions.
"""
from deeptrace.vision import read_static_image
from deeptrace import face_analyser, state_manager
import os

def init_state():
    state_manager.init_item('face_detector_model', 'yolo_face')
    state_manager.init_item('face_detector_size', '640x640')
    state_manager.init_item('face_detector_score', 0.5)
    state_manager.init_item('face_detector_angles', [0])

def main():
    init_state()
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    files = sorted([f for f in os.listdir(uploads_dir) if f.startswith('source_')])
    if not files:
        print('No source_ files found in uploads')
        return

    for fname in files:
        path = os.path.join(uploads_dir, fname)
        print('\n==>', fname)
        img = read_static_image(path)
        if img is None:
            print('  Unable to read image')
            continue
        faces = face_analyser.get_many_faces([img])
        print('  faces found:', len(faces))
        for i, f in enumerate(faces):
            detector = f.score_set.get('detector') if hasattr(f, 'score_set') else 'N/A'
            landmarker = f.score_set.get('landmarker') if hasattr(f, 'score_set') else 'N/A'
            print(f'   - face {i}: detector={detector}, landmarker={landmarker}')

if __name__ == '__main__':
    main()
