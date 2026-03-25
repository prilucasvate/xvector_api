import random
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import soundfile as sf
import torchaudio
import os
import csv
import time
# 引入新版模組
from dataset import SpeakerDataset, DynamicCollate
from model import XVector

# ==========================================
# 1. 參數設定 
# ==========================================
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
JSONL_PATH = "data/manifest.jsonl"  
run_id = time.strftime("%Y%m%d_%H%M%S")
MODEL_SAVE_PATH = f"best_model_{run_id}.pth"
LOG_FILE = f"train_log_{run_id}.csv"
MUSAN_PATH = "data/musan"
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4)) # 預設 4

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 動態算 EER 的雷達函數
# ==========================================
def evaluate_eer(model, test_jsonl, num_spks_to_test=20, utts_per_spk=5):
    """每個 Epoch 結束時，隨機抽幾個人算 EER 當作真實的指標"""

    was_training = model.training # 記住模型進來時的狀態
    model.eval()                  # 切換到評估模式
    
    # 1. 讀取並整理 Test_Closed 的資料 
    test_data = defaultdict(list)
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('split') == 'test_closed':
                test_data[item['spk_id']].append(item['path'])
                
    # 2. 隨機抽人與抽句子
    spk_ids = random.sample(list(test_data.keys()), min(num_spks_to_test, len(test_data)))
    
    # 我們需要用到 DataLoader 裡讀取和 RMS 的邏輯，所以實例化一個輕量版的 collate
    collate = DynamicCollate(augment=False) 
    
    embeddings = {}
    with torch.no_grad():
        for spk in spk_ids:
            utts = test_data[spk]
            selected_utts = random.sample(utts, min(utts_per_spk, len(utts)))
            
            vecs = []
            for path in selected_utts:
                # 這裡手動讀取音檔並過 RMS (與推論邏輯一致)
                data, sr = sf.read(path)
                wav = torch.from_numpy(data).float()
                if wav.ndim == 1: wav = wav.unsqueeze(0)
                else: wav = wav.transpose(0, 1)
                if sr != 16000:
                    wav = torchaudio.transforms.Resample(sr, 16000)(wav)
                if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
                wav = wav.squeeze(0)
                
                # 關鍵：加上 RMS 標準化
                wav = collate.normalize_rms(wav)
                
                # 丟進模型抽特徵
                wav = wav.unsqueeze(0).to(DEVICE)
                _, embedding = model(wav)
                
                # 轉 numpy 並 L2 正規化
                vec = embedding.cpu().numpy()[0]
                vec = vec / np.linalg.norm(vec)
                vecs.append(vec)
                
            if len(vecs) >= 2:
                embeddings[spk] = vecs

    # 3. 產生配對並算 Cosine Similarity
    labels = []
    scores = []
    spk_list = list(embeddings.keys())
    
    # Positive pairs
    for spk in spk_list:
        vecs = embeddings[spk]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = np.dot(vecs[i], vecs[j])
                scores.append(sim)
                labels.append(1)
                
    # Negative pairs
    for i in range(len(spk_list)):
        for j in range(i + 1, len(spk_list)):
            spk1, spk2 = spk_list[i], spk_list[j]
            vec1 = random.choice(embeddings[spk1])
            vec2 = random.choice(embeddings[spk2])
            sim = np.dot(vec1, vec2)
            scores.append(sim)
            labels.append(0)


    if was_training:
        model.train() # 恢復訓練模式

    # 4. 計算 EER
    if not labels: # 如果資料不夠防呆
        return 100.0
        
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr))) # 找到 FPR 和 FNR 最接近的點
    eer = fpr[eer_index] # 這個點的 FPR 就是 EER
    
    return eer * 100.0 # 回傳百分比

def train():
    # 1. CSV title for logging
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'train_acc', 'EER', 'lr'])


    # ==========================================
    # 2. 準備資料 
    # ==========================================
    print(f"[Train] Loading data from {JSONL_PATH}...")
    
    # 建立訓練集與測試集
    # 在 dataloader.py 會根據 split 篩選
    train_dataset = SpeakerDataset(JSONL_PATH, split='train')
    # test_dataset = SpeakerDataset(JSONL_PATH, split='test_closed')  # use eer

    # DataLoader 把資料打包成 Batch
    # 加入 collate_fn=collate_fn_dynamic
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        collate_fn=DynamicCollate(augment=True, musan_path=MUSAN_PATH) # 訓練做增強
    )
    
    
    

    # 從 Dataset 取得語者總數 
    num_classes = train_dataset.num_classes
    print(f"[Train] Total Speakers (Classes): {num_classes}")

    # ==========================================
    # 3. 初始化模型、損失函數與優化器
    # ==========================================
    model = XVector(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # ArcFace 已經嚴格，不再 label_smoothing
    # Adam + Weight Decay: 防止死背訓練集
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= 0.0001) #1e-4
    # ReduceLR: 當測試卡住時 自動降低LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    best_eer = float('inf') # 初始設為無限大，之後只要比它小就存檔

    # ==========================================
    # 4. 訓練
    # ==========================================
    for epoch in range(NUM_EPOCHS):
        model.train() # 訓練模式
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            # waveforms shape: [Batch, time], labels shape: [Batch]
            waveforms, labels = waveforms.to(DEVICE), labels.to(DEVICE)

            # 前向傳播
            # 前向傳播必須傳入 labels 給 ArcFace 計算 Margin
            logits, _ = model(waveforms, labels) # 直接丟 waveform 進去
            loss = criterion(logits, labels) # 計算loss

            # 反向傳播
            optimizer.zero_grad() # 清空之前的梯度
            loss.backward() # 計算新的梯度
            optimizer.step() # 更新模型參數

            # 統計準確度
            running_loss += loss.item() # 累加 loss
            _, predicted = torch.max(logits, 1) # 最高分數的index
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # 累加正確預測的數量

            # 每 20 個 batch 顯示一次訓練狀態
            if (batch_idx + 1) % 20 == 0: 
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # --- 計算這一個 Epoch 的平均數據 ---
        avg_loss = running_loss / len(train_loader) # 平均 Loss
        train_acc = 100 * correct / total # 計算訓練準確率
        
        # ==========================================
        #  EER 驗證 (代替原本的 Test Acc)
        # ==========================================
        print("Calculating Validation EER...")
        # 每次抽 20 個人，每個人抽 5 句話
        val_eer = evaluate_eer(model, JSONL_PATH, num_spks_to_test=50, utts_per_spk=5)
        
        scheduler.step(val_eer) # 根據 EER 決定要不要降學習率

        is_best = False
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            is_best = True

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | EER: {val_eer:.2f}% | Best EER: {best_eer:.2f}% | LR: {current_lr}")

        if is_best:
            print(f"!!! New Best Model Saved with EER: {best_eer:.2f}% !!!\n")

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, train_acc, val_eer, current_lr])

    print(f"\nfinish! lowest EER is: {best_eer:.2f}%")

if __name__ == "__main__":
    train()