import os
import sys
import io
import torch
import torchaudio
import numpy as np
import soundfile as sf

# 將上一層的 train 目錄加入系統路徑，才能匯入 model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(current_dir, '..', 'train')
sys.path.append(train_dir)

from se_model import XVector

class SpeakerEncoder:
    def __init__(self, model_path, num_classes, input_dim=80, cohort_path=None):
        """
        初始化 X-vector 特徵提取器。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        self.cohort_matrix = None
        print(f"[Inference] Initializing model on {self.device} ...")
        
        # 初始化模型並載入權重
        self.model = XVector(num_classes=num_classes, input_dim=input_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        # 切換到推論模式，關閉 Dropout 與 ReLU
        self.model.eval() 
        print(f"[Inference] Model state : {'Training' if self.model.training else 'Eval (No ReLU)'}")
        if cohort_path:
            self.cohort_matrix = np.load(cohort_path)
            cohort_norms = np.linalg.norm(self.cohort_matrix, axis=1, keepdims=True)
            self.cohort_matrix = self.cohort_matrix / np.maximum(cohort_norms, 1e-12)
            print(f"[Inference] Cohort matrix loaded: {self.cohort_matrix.shape}")

    def preprocess_audio(self, audio_bytes):
        """
        接收記憶體中的音檔位元組 (Bytes)，並執行與訓練時相同的 DSP 前處理。
        """
        # 使用 io.BytesIO 在記憶體中直接讀取音檔，不碰硬碟 I/O
        data, sr = sf.read(io.BytesIO(audio_bytes))
        waveform = torch.from_numpy(data).float()
        
        # 1. 單聲道處理
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
            
        # 2. Resampling (確保 16kHz)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # 3. 雙聲道平均化
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0) # 變回 [Samples] 維度

        # 4. 嚴格對齊訓練集的 RMS 能量標準化
        target_rms = 0.05
        rms = torch.sqrt(torch.mean(waveform**2))
        if rms > 0:
            waveform = waveform * (target_rms / rms)
        waveform = torch.clamp(waveform, min=-1.0, max=1.0)
            
        return waveform

    def extract_embedding(self, audio_bytes):
        """
        傳入音檔位元組，回傳 512 維並經過 L2 正規化的特徵陣列。
        """
        # 1. 前處理
        waveform = self.preprocess_audio(audio_bytes)
        waveform = waveform.unsqueeze(0).to(self.device) # 增加 Batch 維度 -> [1, Samples]
        
        # 2. 模型推論
        with torch.no_grad():
            _, embedding = self.model(waveform)
            
        # 3. 轉為 Numpy Array
        vec = embedding.cpu().numpy()[0]
        
        # 4. L2 正規化 (讓 Cosine Similarity 更精準)
        vec = vec / np.linalg.norm(vec)
        
        return vec # 回傳 512 維的特徵向量 

    def score(self, vec1, vec2):
        raw_score = float(np.dot(vec1, vec2))
        if self.cohort_matrix is None:
            return {
                "raw_score": raw_score,
                "snorm_score": None,
            }

        scores1 = np.dot(self.cohort_matrix, vec1)
        scores2 = np.dot(self.cohort_matrix, vec2)
        mean1, std1 = np.mean(scores1), np.std(scores1)
        mean2, std2 = np.mean(scores2), np.std(scores2)
        norm_score1 = (raw_score - mean1) / (std1 + 1e-6)
        norm_score2 = (raw_score - mean2) / (std2 + 1e-6)

        return {
            "raw_score": raw_score,
            "snorm_score": float((norm_score1 + norm_score2) / 2.0),
        }