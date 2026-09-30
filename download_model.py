"""Download the official MediaPipe FaceLandmarker task with integrity verification."""
from pathlib import Path
import hashlib
import urllib.request

URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
SHA256 = '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'

def main():
    target = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'
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
