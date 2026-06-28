import os
import random
import warnings
import json

import torch
import torchaudio
import torchaudio.functional as F_audio  
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path

warnings.filterwarnings("ignore")

# ==========================================
# 所有音檔處理取樣率
SAMPLE_RATE = 16000  

class SpeakerDataset(Dataset):
    """
    負責讀取 JSONL 格式的資料清單，並建立語者 ID 到數字 Label 的對應表。
    確保訓練與測試集使用的是同一個類別字典。
    """
    def __init__(self, jsonl_path, split='train', data_root=None):
        self.data_root = Path(data_root) if data_root else None
        self.data = [] 
        self.labels = [] 

        all_raw_items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f: 
                all_raw_items.append(json.loads(line.strip()))
        
        # 建立全域的語者對應表 (Dictionary)
        # 必須確保 train 看到相同的 num_classes，
        # 否則分類層 (ArcFace) 的維度會對不齊。
        valid_spks = set()
        for item in all_raw_items:
            if item['split'] in ['train']:
                valid_spks.add(item['spk_id'])
                
        all_spks = sorted(list(valid_spks))
        self.spk2idx = {spk: i for i, spk in enumerate(all_spks)}
        self.num_classes = len(all_spks)

        # 根據外部指定的 split (如 'train', 'val', 'test') 篩選資料
        for item in all_raw_items:
            if item['split'] == split:
                self.data.append(item)

        print(f"[Dataset] {split} set loaded. Dict Classes: {self.num_classes}, Samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        wav_path = Path(item["path"])
        if self.data_root is not None and not wav_path.is_absolute():
            wav_path = self.data_root / wav_path

        # 若語者不在字典內，給 -1 避免 Key Error 導致訓練中斷
        label = self.spk2idx.get(item['spk_id'], -1)    

        # 讀取音檔波形
        data, sr = sf.read(str(wav_path))
        waveform = torch.from_numpy(data).float()
        
        # 正規化維度為 [Channels, Samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
        
        # 確保取樣率絕對是 16kHz
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # 多聲道降為單聲道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 回傳維度: (Samples,) 1D Tensor 與對應的整數標籤
        return waveform.squeeze(0), label

# =========================================================================
class DynamicCollate:
    """
    自定義 Batch 處理 (Collate Function)。
    1. 動態長度：整個 Batch 使用同一個隨機長度(2-4秒)。
    2. 資料增強：在 RAM 中高速疊加 MUSAN 噪音與 RIR 殘響，避免硬碟 I/O 瓶頸。
    """
    def __init__(self, augment=False, musan_path=None, rir_path=None):
        self.augment = augment
        self.musan_tensors = []
        self.rir_tensors = []

        # 預先將所有噪音檔案的路徑載入記憶體，避免訓練時頻繁掃描資料夾
        # 將 MUSAN 噪音讀取為 Tensor 並存入 RAM (徹底消滅硬碟 I/O)
        if self.augment and musan_path and os.path.exists(musan_path):
            print(f"Loading MUSAN noise library to RAM: {musan_path} ...")
            for root, dirs, files in os.walk(musan_path):
                for file in files:
                    if file.endswith('.wav'):
                        full_path = os.path.join(root, file)
                        try:
                            noise_data, sr = sf.read(full_path)
                            noise_tensor = torch.from_numpy(noise_data).float()
                            if noise_tensor.ndim > 1:
                                noise_tensor = noise_tensor.mean(dim=1)
                            if sr != SAMPLE_RATE:
                                noise_tensor = F_audio.resample(noise_tensor, orig_freq=sr, new_freq=SAMPLE_RATE)
                            self.musan_tensors.append(noise_tensor)
                        except Exception:
                            pass
            print(f"Successfully loaded {len(self.musan_tensors)} MUSAN Tensors!")
            
        # 載入預先處理好的 RIR 
        if self.augment and rir_path and os.path.exists(rir_path):
            print(f"Loading preprocessed RIR cache: {rir_path}")
            payload = torch.load(rir_path, map_location="cpu")
            self.rir_tensors = payload["rir_tensors"]
            print(f"Successfully loaded {len(self.rir_tensors)} preprocessed RIRs")

    def normalize_rms(self, wav, target_rms=0.05):
        """
        音量標準化。
        錄音設備與環境不同會導致音量落差。使用 RMS 能真實反映人耳感受到的能量，
        避免特定極大音量的樣本主導神經網路的權重。
        """
        rms = torch.sqrt(torch.mean(wav**2))
        if rms > 0:
            wav = wav * (target_rms / rms)
            
        # 防止 RMS 放大後，峰值超出 [-1.0, 1.0] 造成爆音 
        return torch.clamp(wav, min=-1.0, max=1.0)


    def __call__(self, batch):
        """
        當 DataLoader 準備好一個 Batch 的資料時，會呼叫此函式進行打包。
        """
        waveforms, labels = zip(*batch) 

        batch_size = len(waveforms)
        # 動態決定此 Batch 的統一長度 (32000 = 2秒, 64000 = 4秒)
        # 這有助於模型學習適應不同長度的輸入，提升泛化能力
        batch_length = random.randint(32000, 64000) 
        
        # =======================================================
        # 1: 長度對齊 (因為每個音檔初始長度不同)
        padded_waveforms = []
        for w in waveforms:
            seq_len = w.shape[0]
            
            # --- 長度對齊處理 ---
            if seq_len > batch_length:
                # 語音過長：隨機抽取其中一段
                start = random.randint(0, seq_len - batch_length)
                w = w[start : start + batch_length]
                
            elif seq_len < batch_length:
                # 語音過短： Looping 直到達到規定長度
                repeats = (batch_length // seq_len) + 1
                w = w.repeat(repeats)
                w = w[:batch_length]
            padded_waveforms.append(w)

        #  把 256 個獨立的 1D Tensor，疊成一個超級大的 2D 矩陣
        # shape 變成: [Batch_size, Time] ([256, 48000])
        # 接下來所有的運算，都直接對這個大矩陣操作
        batch_wavs = torch.stack(padded_waveforms)
        # =======================================================
        # 步驟 2: 矩陣資料增強 (Data Augmentation)
        if self.augment:
            # 有 70% 的機率，把這一整個 Batch 的聲音套上一層隨機的麥克風濾鏡
            if random.random() < 0.7:
                center_freq = random.uniform(100, 6000) # 從低頻到高頻隨機挑
                gain = random.uniform(-10.0, 10.0)      # 隨機大聲或小聲 10dB
                Q = random.uniform(0.5, 2.0)            # 影響的頻率範圍寬度
                
                # 直接對整個 2D 大矩陣 [Batch, Time] 進行 Biquad 濾波
                batch_wavs = F_audio.equalizer_biquad(batch_wavs, SAMPLE_RATE, center_freq, gain, Q)

            
            # --- 加殘響 (RIR) ---
            if len(self.rir_tensors) > 0:
                # 1. 從 RAM 裡隨機抽 1 個殘響 (這個 Batch 共用同一個殘響環境)
                current_rir = random.choice(self.rir_tensors)
                
                if current_rir.numel() > 1:
                    # 2. 產生 Mask ：這是一個 [256, 1] 的矩陣，裡面只有 0 和 1
                    # 代表 256 個音檔中，大約有 30% 的位置是 1 (要加殘響)，70% 是 0 (保持原聲)
                    reverb_mask = (torch.rand(batch_size) < 0.3).float().unsqueeze(1)
                    
                    # 3. 擴展 RIR 矩陣，讓它變成 [256, RIR長度]
                    rir_2d = current_rir.unsqueeze(0).expand(batch_size, -1)
                    
                    # 4. 高速卷積 直接把 [256, 48000] 的聲音矩陣，跟 [256, RIR長度] 的殘響矩陣做 FFT！
                    reverb_batch = F_audio.fftconvolve(batch_wavs, rir_2d)
                    reverb_batch = reverb_batch[:, :batch_wavs.shape[1]] # 裁切回原本長度
                    
                    # 5. 套用 Mask：沒抽中的 (1-mask) 保持原聲，抽中的 (mask) 換成殘響聲
                    batch_wavs = batch_wavs * (1 - reverb_mask) + reverb_batch * reverb_mask

            # --- 加噪音 (MUSAN) ---
            # 這裡的 self.musan_tensors 是在 __init__ 已經讀進 RAM 的 Tensor 列表
            if len(self.musan_tensors) > 0:
                # 1. 從 RAM 抽 1 個噪音
                current_noise = random.choice(self.musan_tensors)
                
                # 2. 把噪音弄到跟 Batch 一樣長
                if current_noise.shape[0] < batch_length:
                    repeats = (batch_length // current_noise.shape[0]) + 1
                    current_noise = current_noise.repeat(repeats)
                start = random.randint(0, current_noise.shape[0] - batch_length)
                noise_crop = current_noise[start : start + batch_length].unsqueeze(0) # 變成 [1, Time]
                
                # 3. 產生噪音的 Mask (50% 機率加噪音)
                noise_mask = (torch.rand(batch_size) < 0.5).float().unsqueeze(1)
                
                # 4. 一次性產生 256 個不同的 SNR (訊噪比) 值
                snr_db = torch.empty(batch_size, 1).uniform_(0.0, 20.0)
                
                # 5. 計算能量
                # dim=1 代表沿著時間軸算能量，算完後 speech_power 是 [256, 1] 的矩陣
                speech_power = batch_wavs.norm(p=2, dim=1, keepdim=True)
                noise_power = noise_crop.norm(p=2)
                
                if noise_power > 0:
                    # 6. 算出 256 個音檔各自需要的 scale (縮放係數)
                    scale = (speech_power / (10 ** (snr_db / 20))) / noise_power
                    # 7. 把噪音加上去
                    batch_wavs = batch_wavs + (noise_crop * scale) * noise_mask

        # =======================================================
        # 步驟 3:  RMS 能量標準化

        target_rms = 0.05
        # 算出 256 個音檔各自的 RMS 能量，結果是 [256, 1]
        rms = torch.sqrt(torch.mean(batch_wavs**2, dim=1, keepdim=True))
        # 避免除以 0 導致錯誤。一次性把 256 個音檔都調整到 target_rms
        batch_wavs = torch.where(rms > 0, batch_wavs * (target_rms / rms), batch_wavs)
        # 強制截斷，防止爆音
        batch_wavs = torch.clamp(batch_wavs, min=-1.0, max=1.0)

        return batch_wavs, torch.tensor(labels)













