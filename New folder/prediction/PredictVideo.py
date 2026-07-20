import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from django.conf import settings
from facenet_pytorch import MTCNN
from transformers import ViTImageProcessor, ViTForImageClassification
import time

# 1. INITIALIZE HARDWARE AND MODELS GLOBALLY
# We load these outside the function so they stay in memory, making videos process much faster
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading Video Deepfake Models on {DEVICE}...")

mtcnn = MTCNN(keep_all=True, device=DEVICE)
processor = ViTImageProcessor.from_pretrained('dima806/deepfake_vs_real_image_detection')
model = ViTForImageClassification.from_pretrained('dima806/deepfake_vs_real_image_detection').to(DEVICE)
model.eval()

def predict_video_deepfake(video_path):
    """
    Spatial Frame Aggregation Pipeline for Video Deepfake Detection.
    Extracts evenly spaced frames, analyzes faces, and uses a voting system for the final verdict.
    """
    # 2. OPEN VIDEO AND CALCULATE FRAME EXTRACTION MATH
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Could not open video file.'}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # ENGINEERING DECISION: We only extract 10 frames to save computation time on local hardware
    num_frames_to_extract = 10
    
    # np.linspace calculates 10 perfectly even intervals across the entire video timeline
    frame_indices = np.linspace(0, total_frames - 1, num_frames_to_extract, dtype=int)

    frame_results = []
    fake_frames_count = 0
    timestamp = int(time.time())

    # 3. LOOP THROUGH THE EXTRACTED TIMELINE
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert OpenCV BGR to RGB and then to PIL Image for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        img_w, img_h = pil_img.size

        # Detect faces in this specific frame
        boxes, probs = mtcnn.detect(pil_img)

        # Skip this frame if no high-confidence faces are found
        if boxes is None or len(boxes) == 0 or probs[0] < 0.90:
            continue

        # To save time in video, we only analyze the most prominent face (usually the first one returned)
        box = boxes[0] 
        
        # --- THE BOUNDING BOX MARGIN FIX ---
        x1, y1, x2, y2 = [int(coord) for coord in box]
        width, height = x2 - x1, y2 - y1
        margin_x, margin_y = int(width * 0.25), int(height * 0.25)
        
        x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        x2, y2 = min(img_w, x2 + margin_x), min(img_h, y2 + margin_y)
        
        cropped_face = pil_img.crop((x1, y1, x2, y2))
        
        # --- THE INFERENCE PIPELINE ---
        inputs = processor(images=cropped_face, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1).squeeze(0)
            
        predicted_class_idx = probabilities.argmax(-1).item()
        raw_label = model.config.id2label[predicted_class_idx].lower()
        
        if "fake" in raw_label:
            prediction_label = "fake"
            fake_prob = probabilities[predicted_class_idx].item()
            real_prob = 1.0 - fake_prob
            fake_frames_count += 1
        else:
            prediction_label = "real"
            real_prob = probabilities[predicted_class_idx].item()
            fake_prob = 1.0 - real_prob

        # SAVE THE FRAME AS EVIDENCE FOR THE UI
        evidence_filename = f"vid_evidence_frame_{idx}_{timestamp}.png"
        evidence_path = os.path.join(settings.BASE_DIR, 'uploaded_images', evidence_filename)
        
        # Draw the bounding box and label on the frame before saving
        box_color = (0, 255, 0) if prediction_label == "real" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, f"{prediction_label.upper()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        cv2.imwrite(evidence_path, frame)

        # Store data for the frontend
        frame_results.append({
            'frame_number': int(idx),
            'label': prediction_label,
            'fake_confidence': round(fake_prob * 100),
            'real_confidence': round(real_prob * 100),
            'evidence_image': evidence_filename
        })

    cap.release()

    # 4. THE VOTING SYSTEM (AGGREGATION)
    if len(frame_results) == 0:
        return {'error': 'No clear faces detected in the video to analyze.'}

    # ENGINEERING DECISION: If >= 30% of the analyzed frames are FAKE, the video is FAKE.
    # Deepfake glitches often only happen for a split second (like an eye blink), so we need high sensitivity.
    threshold = 0.30 
    fake_ratio = fake_frames_count / len(frame_results)
    
    overall_prediction = "fake" if fake_ratio >= threshold else "real"

    return {
        'prediction': overall_prediction,
        'total_analyzed_frames': len(frame_results),
        'fake_frames_detected': fake_frames_count,
        'fake_ratio_percentage': round(fake_ratio * 100),
        'frame_data': frame_results
    }