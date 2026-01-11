"""
Configuration file for optimized face swap application
Adjust these settings for your specific needs
"""

# ===== APPLICATION SETTINGS =====
APP_HOST = '0.0.0.0'
APP_PORT = 5000
APP_DEBUG = False
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

# ===== DIRECTORY SETTINGS =====
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
TEMP_FOLDER = '.temp'
MODELS_FOLDER = '.assets/models'

# ===== MODEL SETTINGS =====

# Face Swapper Model
# Options: 'inswapper_128', 'inswapper_128_fp16', 'ghost_1_256', 'ghost_2_256', 
#          'ghost_3_256', 'blendswap_256', 'simswap_256', 'hyperswap_1a_256',
#          'hyperswap_1b_256', 'hyperswap_1c_256'
# Recommended: 'inswapper_128_fp16' for speed, 'hyperswap_1b_256' for quality
FACE_SWAPPER_MODEL = 'inswapper_128_fp16'

# Face Detector Model
# Options: 'retinaface', 'scrfd', 'yolo_face', 'yunet'
# Recommended: 'yolo_face' for speed, 'retinaface' for accuracy
FACE_DETECTOR_MODEL = 'yolo_face'

# Face Recognizer Model
# Options: 'arcface_inswapper', 'arcface_simswap', 'arcface_ghost'
FACE_RECOGNIZER_MODEL = 'arcface_inswapper'

# Face Landmarker Model
# Options: '2dfan4', 'peppa_wutz'
FACE_LANDMARKER_MODEL = '2dfan4'

# ===== DETECTION SETTINGS =====

# Face detector size (larger = more accurate but slower)
# Options: '320x320', '640x640', '960x960', '1280x1280'
FACE_DETECTOR_SIZE = '640x640'

# Face detector confidence threshold (0.0-1.0)
# Lower = detect more faces (including false positives)
# Higher = only detect clear faces
FACE_DETECTOR_SCORE = 0.5

# Face recognizer similarity threshold (0.0-1.0)
# Lower = more lenient matching
# Higher = stricter matching
FACE_RECOGNIZER_SIMILARITY = 0.6

# ===== EXECUTION SETTINGS =====

# Execution providers (in order of preference)
# For NVIDIA GPU: ['CUDAExecutionProvider', 'CPUExecutionProvider']
# For CPU only: ['CPUExecutionProvider']
# For AMD GPU: ['ROCMExecutionProvider', 'CPUExecutionProvider']
# For Intel GPU: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
EXECUTION_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# GPU device IDs to use (0 for first GPU, [0,1] for multi-GPU)
EXECUTION_DEVICE_IDS = [0]

# Number of threads for inference
# 0 = auto-detect
# Recommended: 4-8 for most systems
INTRA_OP_NUM_THREADS = 4
INTER_OP_NUM_THREADS = 4

# ===== PERFORMANCE SETTINGS =====

# Enable face detection caching
ENABLE_FACE_CACHE = True

# Face cache size (number of frames to cache)
FACE_CACHE_SIZE = 50

# Batch size for processing multiple frames
BATCH_SIZE = 8

# Number of threads for parallel processing
NUM_THREADS = 4

# Enable memory optimization
ENABLE_MEMORY_OPTIMIZATION = True

# System memory limit in GB (0 = no limit)
SYSTEM_MEMORY_LIMIT = 0

# ===== QUALITY SETTINGS =====

# Enable enhanced preprocessing
ENABLE_ENHANCED_PREPROCESSING = True

# Enable CLAHE (Contrast Limited Adaptive Histogram Equalization)
ENABLE_CLAHE = True

# Enable histogram matching
ENABLE_HISTOGRAM_MATCHING = True

# Enable multi-band blending
ENABLE_MULTIBAND_BLENDING = True

# Blending feather amount (0.0-1.0)
BLENDING_FEATHER = 0.3

# Detail preservation amount (0.0-1.0)
DETAIL_PRESERVATION = 0.95

# ===== OUTPUT SETTINGS =====

# Output image quality (1-100)
OUTPUT_IMAGE_QUALITY = 95

# Output image format
# Options: 'jpg', 'png', 'webp'
OUTPUT_IMAGE_FORMAT = 'png'

# Output video codec
# Options: 'libx264', 'libx265', 'libvpx-vp9'
OUTPUT_VIDEO_CODEC = 'libx264'

# Output video quality (0-51, lower is better)
OUTPUT_VIDEO_QUALITY = 18

# ===== LOGGING SETTINGS =====

# Log level
# Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = 'INFO'

# Enable performance logging
ENABLE_PERFORMANCE_LOGGING = True

# ===== ADVANCED SETTINGS =====

# Enable model quantization (INT8)
# Requires onnxruntime-tools
ENABLE_MODEL_QUANTIZATION = False

# Enable TensorRT optimization (NVIDIA GPUs only)
ENABLE_TENSORRT = False

# Enable OpenVINO optimization (Intel hardware)
ENABLE_OPENVINO = False

# Enable multi-scale processing
ENABLE_MULTISCALE_PROCESSING = False

# Scales for multi-scale processing
MULTISCALE_FACTORS = [1.0, 0.75, 1.25]

# ===== SAFETY SETTINGS =====

# Enable content analysis
ENABLE_CONTENT_ANALYSIS = False

# Maximum face age for processing (-1 = no limit)
MAX_FACE_AGE = -1

# Minimum face age for processing (-1 = no limit)
MIN_FACE_AGE = -1

# Allowed genders (empty list = all)
# Options: ['male', 'female']
ALLOWED_GENDERS = []

# ===== FILE SETTINGS =====

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Allowed video extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Maximum image dimensions
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096

# Maximum video duration in seconds (0 = no limit)
MAX_VIDEO_DURATION = 0

# ===== DEVELOPMENT SETTINGS =====

# Enable Flask debug mode (DO NOT use in production)
FLASK_DEBUG = False

# Enable CORS (for API access)
ENABLE_CORS = False

# API rate limiting (requests per minute, 0 = no limit)
API_RATE_LIMIT = 0
