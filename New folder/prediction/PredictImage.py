import os
import cv2
import time
import torch
import torch.nn.functional as F
from PIL import Image as pImage
from django.conf import settings
from django.shortcuts import render
from facenet_pytorch import MTCNN
from transformers import ViTImageProcessor, ViTForImageClassification

# =========================================================================
# PRODUCTION ENGINE INITIALIZATION (SINGLE MODEL)
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Booting DeepTrace Production Engine on: {DEVICE}")

mtcnn = MTCNN(keep_all=True, device=DEVICE)

# Lock in the stable, modern model
print("Loading Primary Classifier (prithivMLmods)...")
model_name = 'prithivMLmods/Deep-Fake-Detector-Model'
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name).to(DEVICE)
model.eval()

def get_fake_probability(outputs):
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    id2label = model.config.id2label
    fake_prob = 0.0
    for idx, label in id2label.items():
        if "fake" in label.lower():
            fake_prob = probabilities[int(idx)].item()
            break
    return fake_prob

def predict_face_deepfake(request):
    if not request.FILES:
        return render(request, 'image.html', {'error': 'No image uploaded.'})
        
    file_key = list(request.FILES.keys())[0]
    uploaded_file = request.FILES[file_key]
    
    timestamp = int(time.time())
    original_file_name = f"original_{uploaded_file.name.split('.')[0]}-{timestamp}.png"
    
    upload_dir = os.path.join(settings.BASE_DIR, 'uploaded_images')
    os.makedirs(upload_dir, exist_ok=True)
    
    image_path = os.path.join(upload_dir, original_file_name)
    
    with open(image_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    image = pImage.open(image_path)
    img_w, img_h = image.size

    boxes, probs = mtcnn.detect(image)
    
    if boxes is None or len(boxes) == 0:
        return render(request, 'image.html', {'error': 'No human faces detected.'})

    open_cv_image = cv2.imread(image_path)
    face_results = []
    
    for i, box in enumerate(boxes):
        if probs[i] < 0.90:
            continue
            
        x1, y1, x2, y2 = [int(coord) for coord in box]
        width = x2 - x1
        height = y2 - y1
        
        # Consistent Portrait Margin (+20%) to retain contextual skin boundaries
        margin_x = int(width * 0.20)
        margin_y = int(height * 0.20)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(img_w, x2 + margin_x)
        y2 = min(img_h, y2 + margin_y)
        
        cropped_face = image.crop((x1, y1, x2, y2)).convert("RGB")
        
        crop_filename = f"crop_face_{i}-{timestamp}.png"
        crop_path = os.path.join(upload_dir, crop_filename)
        cropped_face.save(crop_path)
        
        # Single inference pass to protect the 4GB hardware limit
        with torch.no_grad():
            inputs = processor(images=cropped_face, return_tensors="pt").to(DEVICE)
            outputs = model(**inputs)
            prob_fake = get_fake_probability(outputs)

        prediction_label = "fake" if prob_fake >= 0.50 else "real"
        prob_real = 1.0 - prob_fake

        threat_level = "Synthetic Media Detected" if prediction_label == "fake" else "Authentic Media"

        face_results.append({
            'label': prediction_label,
            'threat': threat_level,
            'real_confidence': round(prob_real * 100),
            'fake_confidence': round(prob_fake * 100),
            'image_name': crop_filename
        })

        box_color = (0, 255, 0) if prediction_label == "real" else (0, 0, 255)
        cv2.rectangle(open_cv_image, (x1, y1), (x2, y2), box_color, 3)
        label_text = f"{prediction_label.upper()} ({max(prob_real, prob_fake)*100:.0f}%)"
        cv2.putText(open_cv_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    if len(face_results) == 0:
        return render(request, 'image.html', {'error': 'No valid faces verified.'})

    processed_file_name = f"processed_{uploaded_file.name.split('.')[0]}-{timestamp}.png"
    processed_file_path = os.path.join(upload_dir, processed_file_name)
    cv2.imwrite(processed_file_path, open_cv_image)

    any_fakes = any(f['label'] == 'fake' for f in face_results)
    overall_result = "fake" if any_fakes else "real"

    from django.apps import apps
    UserImage = None
    for app_config in apps.get_app_configs():
        try:
            UserImage = app_config.get_model('UserImage')
            break
        except LookupError:
            continue
            
    if UserImage:
        try:
            UserImage.objects.create(
                user=request.user if request.user.is_authenticated else None,
                original_image=original_file_name,
                processed_image=processed_file_name,
                result=overall_result,
                fake_prediction=face_results[0]['fake_confidence'],
                real_prediction=face_results[0]['real_confidence']
            )
        except Exception:
            pass

    context = {
        'prediction': overall_result,
        'face_count': len(face_results),
        'all_faces': face_results,
        'processed_image_name': processed_file_name
    }
    
    return render(request, 'image.html', context)