import os
import torch
import torchaudio
import json
import random
from torch.utils.data import Dataset
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")


SAMPLE_RATE = 16000  # 取樣率 16k

class SpeakerDataset(Dataset):
    def __init__(self, jsonl_path, split='train'): # 初始化，讀取 train/test 的資料
        self.data = [] 
        self.labels = [] 

        all_raw_items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f: 
                all_raw_items.append(json.loads(line.strip()))
        
        # 只提取合法語者 (train 和 test_closed) 來建立分類層
        valid_spks = set()
        for item in all_raw_items:
            if item['split'] in ['train', 'test_closed']:
                valid_spks.add(item['spk_id'])
                
        # 建立的全域語者字典
        all_spks = sorted(list(valid_spks))
        self.spk2idx = {spk: i for i, spk in enumerate(all_spks)}
        self.num_classes = len(all_spks)

        # 根據我們需要的 split，把對應的資料裝進 self.data
        for item in all_raw_items:
            if item['split'] == split:
                self.data.append(item)

        print(f"[Dataset] {split} set loaded. Dict Classes: {self.num_classes}, Samples: {len(self.data)}")

    # 3. 定義 __len__ 和 __getitem__  給 DataLoader 使用
    def __len__(self):
        return len(self.data) # 回傳資料集的總樣本數量

    def __getitem__(self, idx):
        item = self.data[idx]
        wav_path = item['path']

        # 如果這筆資料的語者不在字典裡 ，直接給予標籤 -1 避免當機
        # 根據 spk_id 取得對應的數字 label
        label = self.spk2idx.get(item['spk_id'], -1)    

        # 如果是 train 或 test_closed 階段，卻找不到 label，直接報錯
        if label == -1 and item['split'] in ['train', 'test_closed']:
            raise ValueError(f"[Dataset] Error : cannot find corresponding ID for speaker {item['spk_id']} ! Please regenerate manifest.jsonl")

        # 3. 讀取音檔 (只讀取原始波形)
        # data 的 shape 通常是 (samples,) 或 (samples, channels)
        data, sr = sf.read(wav_path)
        
        # 將 numpy 轉為 torch tensor 並轉置維度
        waveform = torch.from_numpy(data).float()
        
        # 確保維度是 [Channels, Samples]，如果是單聲道 [Samples] -> [1, Samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)
        
        # 確保採樣率是 16k
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # 如果是多聲道，取平均成單聲道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 改掉統一 3 秒 

        # 擠壓掉 channel 維度 回傳 (samples,) 的一維 tensor 和 label
        # [1, Samples] -> [Samples]
        return waveform.squeeze(0), label


# 動態長度和音檔循環
# ====================================================
class DynamicCollate:
    """
    負責動態裁切長度、音量標準化，以及掛載 MUSAN 噪音
    """
    def __init__(self, augment=False, musan_path=None):
        self.augment = augment
        self.musan_wavs = []

        # 掃描並快取所有 MUSAN 噪音檔的路徑
        if self.augment:
            if musan_path and os.path.exists(musan_path):
                print(f"[Dataset] Scanning MUSAN noise library: {musan_path} ...")
                for root, dirs, files in os.walk(musan_path):
                    for file in files:
                        if file.endswith('.wav'):
                            self.musan_wavs.append(os.path.join(root, file))
                print(f"[Dataset] Successfully loaded {len(self.musan_wavs)} noise files for augmentation!")
            else:
                # 開啟了增強，找不到檔案
                print(f"[Dataset] Warning: augment=True is enabled but MUSAN directory '{musan_path}' not found! Training will be conducted without noise augmentation!")



    def normalize_rms(self, wav, target_rms=0.05):
        """
        使用 RMS 進行能量標準化。
        解決突發大聲音導致整體音量被過度壓縮。
        """
        # 計算整段音檔的平均能量
        rms = torch.sqrt(torch.mean(wav**2))
        if rms > 0:
            wav = wav * (target_rms / rms)
            
        # 加上 clamp 確保即使放大後，瞬間突波也不會超過數值極限 [-1.0, 1.0] 導致爆音
        return torch.clamp(wav, min=-1.0, max=1.0)

    def fast_add_noise(self, speech, noise_tensor):
        """完全在記憶體 (RAM) 內執行的高速混音，不碰硬碟"""
        # 噪音不夠長就重複疊加
        if noise_tensor.shape[0] < speech.shape[0]:
            repeats = (speech.shape[0] // noise_tensor.shape[0]) + 1
            noise_tensor = noise_tensor.repeat(repeats)
            
        # 隨機裁切一段出來 (每個音檔切的位置都不一樣，確保多樣性)
        start = random.randint(0, noise_tensor.shape[0] - speech.shape[0])
        noise_crop = noise_tensor[start : start + speech.shape[0]]
        
        # 隨機決定 SNR (5dB 到 20dB)
        snr_db = random.uniform(5.0, 20.0)
        
        speech_power = speech.norm(p=2)
        noise_power = noise_crop.norm(p=2)
        
        if noise_power == 0:
            return speech
            
        scale = (speech_power / (10 ** (snr_db / 20))) / noise_power
        return speech + noise_crop * scale

    def __call__(self, batch):
        waveforms, labels = zip(*batch) # 解壓 batch 中的 waveforms 和 labels，得到兩個 tuple
        batch_length = random.randint(32000, 64000) # 決定這個 Batch 的動態長度 (2~4秒)
        
        # 這個 Batch 只去硬碟讀1次噪音！
        # ==========================================
        current_batch_noise = None
        if self.augment and len(self.musan_wavs) > 0:
            noise_path = random.choice(self.musan_wavs)
            try:
                noise_data, sr = sf.read(noise_path)
                noise_tensor = torch.from_numpy(noise_data).float()
                if noise_tensor.ndim > 1:
                    noise_tensor = noise_tensor.mean(dim=1)
                # 如果真的遇到非 16k，用 functional 輕量轉頻
                if sr != 16000:
                    import torchaudio.functional as F_audio
                    noise_tensor = F_audio.resample(noise_tensor, orig_freq=sr, new_freq=16000)
                current_batch_noise = noise_tensor
            except Exception:
                pass # 如果剛好抽到壞檔就算了，這個 batch 就不加噪音

        padded_waveforms = []
        for w in waveforms:
            seq_len = w.shape[0]
            
            # 2. 太長：隨機起點裁切
            if seq_len > batch_length:
                start = random.randint(0, seq_len - batch_length)
                w = w[start : start + batch_length]
                
            # 3. 太短：循環重複 (Looping) 直到滿足長度
            elif seq_len < batch_length:
                # 算出需要重複幾次才夠長 (例如 1.5 秒要塞滿 4 秒，就重複 3 次)
                repeats = (batch_length // seq_len) + 1
                w = w.repeat(repeats)
                # 然後裁切到我們要的長度
                w = w[:batch_length]
                
            # 1. 語音能量標準化 (RMS)   # 其實放在最後加噪音後再做一次標準化會更合理，確保不會因為加噪音而爆音
            # w = self.normalize_rms(w)
                
            # 2. 加入背景噪音 (50% 機率)
            if current_batch_noise is not None and random.random() < 0.5:
                # 傳入已經在記憶體裡的 Tensor 進行高速混音
                w = self.fast_add_noise(w, current_batch_noise)

            w = self.normalize_rms(w) # 全部在這裡做唯一一次的 RMS=0.05
                    
            padded_waveforms.append(w)
            
        return torch.stack(padded_waveforms), torch.tensor(labels)
