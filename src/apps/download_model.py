"""Download the official MediaPipe FaceLandmarker task with integrity verification."""

# --- 路径引导：src/ 下 core / pipeline / experiments / apps 之间可互相 import ---
# 本段由结构重构引入。算法逻辑不依赖它，仅用于把同级子目录加入模块搜索路径，
# 使 `from feature_extractor import ...` 这类平铺导入在跨目录后依然有效。
import os as _os
import sys as _sys

_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ("core", "pipeline", "experiments", "apps"):
    _sub_path = _os.path.join(_SRC_DIR, _sub)
    if _os.path.isdir(_sub_path) and _sub_path not in _sys.path:
        _sys.path.insert(0, _sub_path)
# --- 路径引导结束 ---

from pathlib import Path
import hashlib
import urllib.request

URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'

def main():
    target = Path(__file__).resolve().parent.parent.parent / 'models' / 'face_landmarker.task'
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print('FaceLandmarker model already installed and verified.')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise RuntimeError('Model checksum mismatch; download was not installed')
    temp = target.with_suffix('.download')
    temp.write_bytes(content)
    temp.replace(target)
    print(f'Installed: {target}')

if __name__ == '__main__':
    main()
