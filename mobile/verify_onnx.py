import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

"""
驗證 ONNX 輸出與原始 PyTorch 模型一致。
比較 Mel 特徵與最終 Embedding，在不同音訊長度下計算 Cosine Similarity。
"""

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from export_onnx import load_xvector, ConvMelSpectrogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join(current_dir, '..', 'api', 'weights', 'best_model_plus_v14.pth'))
    parser.add_argument('--onnx', default=os.path.join(current_dir, 'xvector_v14.onnx'))
    parser.add_argument('--threshold', type=float, default=0.9999)
    args = parser.parse_args()

    model = load_xvector(args.model)
    conv_mel = ConvMelSpectrogram(model.mel_transform).eval()
    session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    torch.manual_seed(0)
    worst = 1.0
    for seconds in [1, 2, 3, 5, 10]:
        wav = torch.randn(1, 16000 * seconds) * 0.05
        with torch.no_grad():
            mel_ref = model.db_transform(model.mel_transform(wav))
            mel_diff = (conv_mel(wav) - mel_ref).abs().max().item()
            _, emb = model(wav)
            ref = F.normalize(emb, dim=1).numpy()[0]
        out = session.run(None, {'waveform': wav.numpy()})[0][0]
        cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out)))
        worst = min(worst, cos)
        print(f"[Verify] {seconds:>2}s  mel max diff = {mel_diff:.2e} dB  cosine = {cos:.6f}")

    passed = worst >= args.threshold
    print(f"[Verify] {'PASS' if passed else 'FAIL'} (worst cosine {worst:.6f}, threshold {args.threshold})")
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
