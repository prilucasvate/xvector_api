import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
將訓練好的 X-Vector 匯出為 ONNX，供手機端離線推論使用。
資料流為 : 原始Waveform [1, Samples] -> 512 維 Embedding (已 L2 正規化)

與 PyTorch 版本的差異：
1. torchaudio 的 MelSpectrogram 內部使用 STFT，部分手機端 Runtime 不支援，
   這裡改用固定 DFT 權重的 Conv1d 實作，數值與 torchaudio 對齊。
2. 移除 ArcFace 分類層，只輸出 Embedding。
3. 輸出前先做 L2 正規化，手機端直接做內積即為 Cosine Similarity。
"""

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'train'))

from model import XVector


class ConvMelSpectrogram(nn.Module):
    """
    以 Conv1d 實作 STFT + Mel 濾波，取代 torchaudio.transforms.MelSpectrogram。
    參數直接從原本的 MelSpectrogram 讀取，確保 window 與 Mel 濾波器完全相同。
    """
    def __init__(self, mel_transform):
        super(ConvMelSpectrogram, self).__init__()
        spec = mel_transform.spectrogram
        self.n_fft = spec.n_fft
        self.hop_length = spec.hop_length

        # torchaudio 會把長度為 win_length 的 window 置中補零到 n_fft
        window = spec.window
        pad_left = (self.n_fft - window.numel()) // 2
        window = F.pad(window, (pad_left, self.n_fft - window.numel() - pad_left))

        # DFT 權重：實部與虛部各一組，[n_freq, 1, n_fft]
        n_freq = self.n_fft // 2 + 1
        n = torch.arange(self.n_fft, dtype=torch.float64)
        k = torch.arange(n_freq, dtype=torch.float64).unsqueeze(1)
        angle = 2 * torch.pi * k * n / self.n_fft
        real = (torch.cos(angle) * window).float().unsqueeze(1)
        imag = (-torch.sin(angle) * window).float().unsqueeze(1)
        self.register_buffer('dft_real', real)
        self.register_buffer('dft_imag', imag)

        # Mel 濾波器：[n_freq, n_mels]
        self.register_buffer('mel_fb', mel_transform.mel_scale.fb.clone())

    def forward(self, x):
        # x: [Batch, Samples]，與 torchaudio 預設相同：center=True, reflect padding
        pad = self.n_fft // 2
        x = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect')
        real = F.conv1d(x, self.dft_real, stride=self.hop_length)
        imag = F.conv1d(x, self.dft_imag, stride=self.hop_length)
        power = real ** 2 + imag ** 2                     # -> [Batch, n_freq, Frames]
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)
        # AmplitudeToDB (power, ref=1.0, amin=1e-10, top_db=None)
        return 10.0 * torch.log10(torch.clamp(mel, min=1e-10))


class XVectorEmbedding(nn.Module):
    """
    推論用包裝：Waveform -> L2 正規化後的 512 維 Embedding。
    """
    def __init__(self, xvector):
        super(XVectorEmbedding, self).__init__()
        self.mel = ConvMelSpectrogram(xvector.mel_transform)
        self.m = xvector

    def forward(self, waveform):
        m = self.m
        x = self.mel(waveform)
        x1 = m.tdnn1(x)
        x2 = m.tdnn2(x1)
        x3 = m.tdnn3(x2)
        x4 = m.tdnn4(x3)
        t = x4.size(2)
        x_mfa = torch.cat((x2[:, :, :t], x3, x4), dim=1)
        x5 = m.tdnn5(x_mfa)
        embedding = m.bn(m.fc(m.asp(x5)))
        return F.normalize(embedding, dim=1)


def load_xvector(model_path):
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    # num_classes 由 ArcFace 權重形狀推得，不需手動指定
    num_classes = state_dict['arcface.weight'].shape[0]
    model = XVector(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join(current_dir, '..', 'api', 'weights', 'best_model_plus_v14.pth'))
    parser.add_argument('--output', default=os.path.join(current_dir, 'xvector_v14.onnx'))
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--cohort', default=os.path.join(current_dir, '..', 'api', 'weights', 'cohort_500_v13.npy'),
                        help='S-Norm cohort (.npy)，設為空字串則不輸出')
    parser.add_argument('--quantize', action='store_true', help='另外輸出 int8 動態量化版本')
    args = parser.parse_args()

    model = XVectorEmbedding(load_xvector(args.model)).eval()
    dummy = torch.randn(1, 16000 * 3) * 0.05

    torch.onnx.export(
        model, (dummy,), args.output,
        input_names=['waveform'],
        output_names=['embedding'],
        dynamic_axes={'waveform': {0: 'batch', 1: 'samples'}, 'embedding': {0: 'batch'}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"[Export] Saved: {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")

    if args.cohort:
        # S-Norm 用的 cohort：L2 正規化後存成 float32 raw binary，[N * 512]，供各平台 SDK 直接讀取
        import numpy as np
        cohort = np.load(args.cohort).astype(np.float32)
        cohort /= np.maximum(np.linalg.norm(cohort, axis=1, keepdims=True), 1e-12)
        c_path = os.path.join(os.path.dirname(args.output), 'cohort.bin')
        cohort.tofile(c_path)
        print(f"[Export] Saved: {c_path} {cohort.shape}")

    if args.quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q_path = args.output.replace('.onnx', '_int8.onnx')
        quantize_dynamic(args.output, q_path, weight_type=QuantType.QInt8)
        print(f"[Export] Saved: {q_path} ({os.path.getsize(q_path) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
