# 手機端離線推論 (ONNX)

將 X-Vector 匯出為 ONNX，可透過 ONNX Runtime Mobile 於 Android / iOS 離線執行。

## 匯出與驗證
```bash
pip install onnx onnxruntime onnxscript
cd mobile
python export_onnx.py --quantize          # 輸出 xvector_v14.onnx 與 xvector_v14_int8.onnx
python verify_onnx.py                     # 比對 PyTorch 與 ONNX 輸出
```
`num_classes` 會從權重自動推得；使用其他版本請加 `--model`、`--output`。

## 模型介面
* 輸入 `waveform`: float32 `[1, Samples]`，16kHz 單聲道，長度可變 (建議 ≥ 1 秒)。
* 輸出 `embedding`: float32 `[1, 512]`，**已 L2 正規化**，兩段音訊的內積即為 Cosine Similarity。

Mel-Spectrogram 已包含在模型內，並以 Conv1d 實作 STFT，因此不需要 Runtime 支援 STFT op。

## App 端前處理
與 `api/inference.py` 相同，需於送入模型前完成：
1. 轉為單聲道 (多聲道取平均)，重新取樣到 16kHz。
2. RMS 標準化到 0.05：`wav = wav * (0.05 / rms)`，再 clamp 到 [-1, 1]。

若需要 S-Norm，請另外打包 `api/weights/cohort_500_v13.npy`，計算方式見 `SpeakerEncoder.score`。

## 驗證結果 (v14)
| 版本 | 大小 | 與 PyTorch 的 Cosine (1–10 秒) |
|---|---|---|
| fp32 | 27.9 MB | 1.000000 |
| int8 (動態量化) | 7.0 MB | 0.982 – 0.991 |

int8 版本以隨機訊號測試有明顯誤差，對 EER 的影響尚未評估，建議先使用 fp32 版本。
