import os
import sys
import json
import numpy as np
import soundfile as sf

"""
用原本的 api/inference.py (PyTorch) 產生參考結果，供 JS SDK 比對。
"""

current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, '..', '..', '..')
sys.path.append(os.path.join(root, 'api'))

from inference import SpeakerEncoder

weights = os.path.join(root, 'api', 'weights')
encoder = SpeakerEncoder(os.path.join(weights, 'best_model_plus_v14.pth'), num_classes=2384,
                         cohort_path=os.path.join(weights, 'cohort_500_v13.npy'))

rng = np.random.default_rng(0)
t = np.arange(16000 * 3) / 16000
golden = {}
for name, f0 in [('a', 120.0), ('b', 210.0)]:
    wav = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * 0.1 + rng.normal(0, 0.01, t.size)
    path = os.path.join(current_dir, f'{name}.wav')
    sf.write(path, wav.astype(np.float32), 16000, subtype='FLOAT')
    with open(path, 'rb') as f:
        golden[name] = encoder.extract_embedding(f.read()).tolist()
golden['score'] = encoder.score(np.array(golden['a']), np.array(golden['b']))
with open(os.path.join(current_dir, 'golden.json'), 'w') as f:
    json.dump(golden, f)
print(golden['score'])
