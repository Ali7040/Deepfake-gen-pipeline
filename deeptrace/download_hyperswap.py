"""Download the hyperswap_1a_256 swapper (native 256px) from facefusion-assets.

hyperswap_1a_256 ships in the models-3.3.0 release. We try GitHub first, then the
HuggingFace mirror. Writes both the .onnx weights and the .hash next to the other
models so the pipeline's hash check passes.
"""
import os
import sys
import urllib.request

DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.assets', 'models')
os.makedirs(DEST_DIR, exist_ok=True)

FILES = ['hyperswap_1a_256.hash', 'hyperswap_1a_256.onnx']
BASES = [
    'https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/{f}',
    'https://huggingface.co/facefusion/models-3.3.0/resolve/main/{f}',
    'https://huggingface.co/facefusion/models/resolve/main/{f}',
]


def download(fname: str) -> bool:
    dest = os.path.join(DEST_DIR, fname)
    min_size = 1_000_000 if fname.endswith('.onnx') else 5
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_size:
        print(f'  [skip] {fname} already present ({os.path.getsize(dest)} bytes)')
        return True
    for base in BASES:
        url = base.format(f=fname)
        try:
            print(f'  [get ] {url}')
            urllib.request.urlretrieve(url, dest)
            if os.path.isfile(dest) and os.path.getsize(dest) >= min_size:
                print(f'  [ ok ] {fname} -> {os.path.getsize(dest)} bytes')
                return True
            os.remove(dest)
        except Exception as e:
            print(f'  [fail] {e}')
            if os.path.isfile(dest):
                try:
                    os.remove(dest)
                except OSError:
                    pass
    return False


ok = all(download(f) for f in FILES)
print('RESULT:', 'SUCCESS' if ok else 'FAILED')
sys.exit(0 if ok else 1)
