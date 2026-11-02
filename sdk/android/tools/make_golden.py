import os
import sys
import json
import numpy as np
import soundfile as sf

"""
用原本的 api/inference.py (PyTorch) 產生 Android instrumentation test 的參考結果。
c.wav 為 44.1kHz 雙聲道，用來驗證 Kotlin 版重新取樣與 torchaudio 一致。
輸出到 xvector/build/golden/。
"""

current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, '..', '..', '..')
out_dir = os.path.join(current_dir, '..', 'xvector', 'build', 'golden')
os.makedirs(out_dir, exist_ok=True)
sys.path.append(os.path.join(root, 'api'))

from inference import SpeakerEncoder

weights = os.path.join(root, 'api', 'weights')
encoder = SpeakerEncoder(os.path.join(weights, 'best_model_plus_v14.pth'), num_classes=2384,
                         cohort_path=os.path.join(weights, 'cohort_500_v13.npy'))


def voice(sr, f0, seconds, rng):
    t = np.arange(int(sr * seconds)) / sr
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * 0.1 + rng.normal(0, 0.01, t.size)


rng = np.random.default_rng(0)
left, right = voice(44100, 120.0, 3, rng), voice(44100, 125.0, 3, rng)
sf.write(os.path.join(out_dir, 'c.wav'), np.stack([left, right], axis=1), 44100, subtype='PCM_16')
sf.write(os.path.join(out_dir, 'd.wav'), voice(16000, 210.0, 3, rng), 16000, subtype='PCM_16')

golden = {}
for name in ['c', 'd']:
    with open(os.path.join(out_dir, f'{name}.wav'), 'rb') as f:
        golden[name] = encoder.extract_embedding(f.read()).tolist()
golden['score'] = encoder.score(np.array(golden['c']), np.array(golden['d']))
with open(os.path.join(out_dir, 'golden.json'), 'w') as f:
    json.dump(golden, f)
print(golden['score'])
