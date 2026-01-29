#!/usr/bin/env python3
"""
Simple Fast Face Swap Application - DeepTrace
Optimized for speed with minimal UI
"""

import os
os.environ['OMP_NUM_THREADS'] = '4'

from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from pathlib import Path
import time
import logging

# Import deeptrace modules
from deeptrace import state_manager, face_detector, face_analyser
from deeptrace.processors.modules.face_swapper import core as face_swapper
import deeptrace.processors.modules.face_enhancer.core as face_enhancer
from deeptrace.vision import read_static_image, write_image
from deeptrace.filesystem import is_image, is_video

# Configure logging - both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deeptrace_app.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary directories
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize DeepTrace state
def initialize_deeptrace():
    """Initialize DeepTrace with optimized settings"""
    try:
        # Download settings - MUST be set first
        state_manager.init_item('download_providers', ['github', 'huggingface'])
        state_manager.init_item('download_scope', 'full')
        state_manager.init_item('log_level', 'info')
        
        # Execution settings
        state_manager.init_item('execution_providers', ['CPUExecutionProvider'])
        state_manager.init_item('execution_device_ids', [0])
        state_manager.init_item('execution_thread_count', 4)
        
        # Face detector settings
        state_manager.init_item('face_detector_model', 'yolo_face')  # Faster detector
        state_manager.init_item('face_detector_size', '640x640')
        state_manager.init_item('face_detector_score', 0.5)
        state_manager.init_item('face_detector_angles', [0])
        state_manager.init_item('face_detector_margin', [0, 0, 0, 0])
        
        # Face swapper settings
        state_manager.init_item('face_swapper_model', 'inswapper_128_fp16')  # Use FP16 for speed
        state_manager.init_item('face_swapper_pixel_boost', '128x128')  # Match model size
        state_manager.init_item('face_swapper_weight', 0.5)
        
        # Face enhancer settings
        state_manager.init_item('face_enhancer_model', 'gfpgan_1.4')
        state_manager.init_item('face_enhancer_blend', 100)
        state_manager.init_item('face_enhancer_weight', 1.0)
        
        # Face recognizer and landmarker
        state_manager.init_item('face_recognizer_model', 'arcface_inswapper')
        state_manager.init_item('face_landmarker_model', '2dfan4')
        state_manager.init_item('face_landmarker_score', 0.5)
        
        # Face mask settings
        state_manager.init_item('face_mask_types', ['box'])
        state_manager.init_item('face_mask_blur', 0.3)
        state_manager.init_item('face_mask_padding', [0, 0, 0, 0])
        state_manager.init_item('face_mask_areas', [])
        state_manager.init_item('face_mask_regions', [])
        state_manager.init_item('face_occluder_model', 'xseg_1')
        state_manager.init_item('face_parser_model', 'bisenet_resnet_34')
        
        # Face selector settings
        state_manager.init_item('face_selector_mode', 'many')
        state_manager.init_item('face_selector_order', 'large-small')
        state_manager.init_item('reference_face_distance', 0.6)
        
        # Output settings
        state_manager.init_item('output_path', app.config['OUTPUT_FOLDER'])
        state_manager.init_item('temp_path', '.temp')
        state_manager.init_item('output_image_quality', 80)
        
        # Video memory strategy
        state_manager.init_item('video_memory_strategy', 'moderate')
        
        logger.info("DeepTrace initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DeepTrace: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# HTML template for simple UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DeepTrace - Fast Face Swap</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .upload-section {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 600;
        }
        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px dashed #667eea;
            border-radius: 8px;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
        }
        input[type="file"]:hover {
            border-color: #764ba2;
            background: #f0f2ff;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 20px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        #status {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            display: none;
        }
        .success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .processing {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }
        #result {
            margin-top: 20px;
            text-align: center;
        }
        #result img {
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .download-btn {
            display: inline-block;
            margin-top: 15px;
            padding: 10px 30px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
        }
        .download-btn:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ DeepTrace v1.1</h1>
        <p class="subtitle">Optimized for Speed & Performance</p>
        
        <form id="uploadForm" method="POST" enctype="multipart/form-data">
            <div class="upload-section">
                <label for="source">Source Face (Person to copy):</label>
                <input type="file" id="source" name="source" accept="image/*" required>
            </div>
            
            <div class="upload-section">
                <label for="target">Target Image/Video (Where to apply):</label>
                <input type="file" id="target" name="target" accept="image/*,video/*" required>
            </div>
            
            <button type="submit" id="submitBtn">Swap Faces</button>
        </form>
        
        <div id="status"></div>
        <div id="result"></div>
    </div>

    <script>
        console.log("DeepTrace App Loaded v1.1");
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            const sourceFile = document.getElementById('source').files[0];
            const targetFile = document.getElementById('target').files[0];
            
            if (!sourceFile || !targetFile) {
                showStatus('Please select both files', 'error');
                return;
            }
            
            formData.append('source', sourceFile);
            formData.append('target', targetFile);
            
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
            
            showStatus('Processing face swap... Please wait.', 'processing');
            
            try {
                const response = await fetch('/swap', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                console.log("Response data:", data);
                
                if (data.success) {
                    showStatus(`Success! Processed in ${data.processing_time.toFixed(2)}s`, 'success');
                    displayResult(data.output_filename || data.output_path);
                } else {
                    showStatus('Error: ' + data.error, 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Swap Faces';
            }
        });
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = type;
            status.style.display = 'block';
        }
        
        function displayResult(outputPath) {
            const result = document.getElementById('result');
            const isVideo = outputPath.endsWith('.mp4') || outputPath.endsWith('.avi');
            const filename = outputPath;
            
            if (isVideo) {
                result.innerHTML = `
                    <video controls style="max-width: 100%; border-radius: 8px;">
                        <source src="/output/${filename}" type="video/mp4">
                    </video>
                    <br>
                    <a href="/output/${filename}" class="download-btn" download>Download Video</a>
                `;
            } else {
                result.innerHTML = `
                    <img src="/output/${filename}" alt="Result">
                    <br>
                    <a href="/output/${filename}" class="download-btn" download>Download Image</a>
                `;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/swap', methods=['POST'])
def swap_faces():
    """Process face swap request"""
    start_time = time.time()
    
    try:
        # Validate files
        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({'success': False, 'error': 'Missing files'}), 400
        
        source_file = request.files['source']
        target_file = request.files['target']
        
        if source_file.filename == '' or target_file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400
        
        if not (allowed_file(source_file.filename) and allowed_file(target_file.filename)):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Save uploaded files
        source_filename = secure_filename(f"source_{int(time.time())}_{source_file.filename}")
        target_filename = secure_filename(f"target_{int(time.time())}_{target_file.filename}")
        
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)
        
        source_file.save(source_path)
        target_file.save(target_path)
        
        # Process face swap
        output_filename = f"output_{int(time.time())}.png"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Perform face swap using DeepTrace
        success = process_face_swap(source_path, target_path, output_path)
        
        processing_time = time.time() - start_time
        
        if success:
            return jsonify({
                'success': True,
                'output_path': output_path,
                'output_filename': output_filename,
                'processing_time': processing_time
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Face swap failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in swap_faces: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/output/<filename>')
def get_output(filename):
    """Serve output file"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

def process_face_swap(source_path, target_path, output_path):
    """
    Perform the actual face swap using DeepTrace
    Optimized for speed and accuracy
    """
    import traceback
    try:
        # Read images
        source_frame = read_static_image(source_path)
        target_frame = read_static_image(target_path)
        
        if source_frame is None or target_frame is None:
            logger.error("Failed to read images")
            return False
        
        # Set source paths for the face swapper
        state_manager.set_item('source_paths', [source_path])
        
        # Get source face
        source_faces = face_analyser.get_many_faces([source_frame])
        if not source_faces:
            logger.error("No face found in source image")
            return False
        
        source_face = source_faces[0]
        logger.info(f"Source face detected with score: {source_face.score if hasattr(source_face, 'score') else 'N/A'}")
        
        # Get target faces
        target_faces = face_analyser.get_many_faces([target_frame])
        if not target_faces:
            logger.error("No face found in target image")
            return False
        
        logger.info(f"Target face detected, count: {len(target_faces)}")
        
        # Use the swap_face function directly instead of process_frame
        # as it's simpler and more direct
        result_frame = target_frame.copy()
        
        for target_face in target_faces:
            # First swap the face
            result_frame = face_swapper.swap_face(source_face, target_face, result_frame)
            # Then enhance the swapped face
            result_frame = face_enhancer.enhance_face(target_face, result_frame)
        
        # Write output
        success = write_image(output_path, result_frame)
        logger.info(f"Output written: {success}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in process_face_swap: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == '__main__':
    # Initialize DeepTrace
    if initialize_deeptrace():
        logger.info("Starting DeepTrace application...")
        # Run on all interfaces, port 5000
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        logger.error("Failed to initialize application")
