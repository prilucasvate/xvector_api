# Speaker Embedding Model V14 - 模型訓練指南

本資料夾包含 Speaker Embedding V14 的訓練與評估流程。

V14 模型使用 **TDNN + SE Block + Attentive Statistics Pooling (ASP) + ArcFace**。此版本加入多層特徵融合與注意力機制，並在訓練過程中使用 ArcFace 損失函數來增強語者辨識能力。
在內部測試設定下，V14 達成約 **5.56% EER**。

## 資料夾結構
* `configs/`：存放訓練與測試的 YAML 設定檔。
* `data/`：放置 sample manifest 或自備資料集。
* `create_manifest.py`：從 data/wavs/<speaker_id>/*.wav 產生 JSONL manifest。
* `model.py`：V14 模型架構。
* `dataset.py`：資料讀取 (支援 MUSAN/RIR 資料增強)。
* `train.py`：訓練主程式。
* `eval.py`：評估主程式。

---
## 1. 環境設置
本專案建議在具備 CUDA 支援的 GPU 下執行。

安裝系統依賴：
```bash
sudo apt-get update
sudo apt-get install libsndfile1
```
安裝 Python 相關套件：
```bash
pip install -r ../requirements.txt
```
## 2. Manifest 格式
訓練與評估皆使用 JSONL manifest。每一行是一筆音檔資料。
必要欄位：
```
{"path":"<音檔路徑>","spk_id":"<語者ID>","split":"<train/val/test>","duration":<音檔秒數>}
```
範例：
```json
{"path":"speaker_01/utt_001.wav","spk_id":"speaker_01","split":"train","duration":3.2}
```
欄位說明：
* path：音檔路徑。可為絕對路徑，或相對於 config 裡的 data_root。
* spk_id：語者 ID。
* split：資料切分。 V14 使用 train、val、test。
* duration：音檔秒數，用於紀錄與檢查。

## 3. 使用自備資料產生 Manifest
請準備您的語音資料夾，架構如下：
```text
data/wavs/  
├── speaker_01/  
│   ├── utt_001.wav  
│   ├── utt_002.wav  
│   └── ...  
└── speaker_02/  
    ├── utt_001.wav  
    ├── utt_002.wav  
    └── ...  
```
* 接著產生訓練用的 JSONL Manifest，這將過濾過短的音檔並切分 Train/Val/Test 集：

```bash
python create_manifest.py --data-dir data/wavs --output data/manifest.jsonl
```
此腳本會：
* 過濾短於 2 秒的音檔。
* 過濾音檔數不足的語者。
* 依 speaker 切分 train / val / test。
* 輸出相對於 data/wavs 的音檔路徑。
注意：create_manifest.py 主要供自備資料集使用。原始 V14 實驗使用固定的 720 小時 manifest，不由此腳本重新產生。

## 4. Config 設定
我們透過 configs/v14.yaml 管理所有參數。請確保設定檔中的 manifest 路徑正確，然後執行：
```bash
python train.py --config configs/v14.yaml
```
* 若設定檔中包含 musan_path 與 rir_path，系統將會在初始化時將噪音庫載入 RAM 中。確保系統有至少 16GB 的記憶體。

* 動態 EER 驗證：每個 Epoch 結束時，系統會自動在驗證集上計算 EER，並保存最佳模型至 outputs/ 目錄。

範例：
```
data:
manifest: data/manifest.jsonl
data_root: data/wavs

output:
dir: outputs/v14

training:
seed: 42
epochs: 100
batch_size: 256
learning_rate: 0.001
num_workers: 4
weight_decay: 0.0001

augmentation:
musan_path: null
rir_path: null

evaluation:
checkpoint: outputs/v14/best_model_plus_v14.pth
split: test
num_spks: 50
utts_per_spk: 10
rounds: 10
```
若 manifest 裡的 path 是相對路徑，例如：
```
{"path":"speaker_01/utt_001.wav","spk_id":"speaker_01","split":"train","duration":3.2}
```
則程式會讀取：
```
"<data_root> / <path>"
```
若 manifest 裡的 path 已經是絕對路徑，則可將 data_root 設為：
```
data_root: null
```

## 5. 模型訓練
確認 configs/v14.yaml 的 manifest 與 data_root 正確後執行：
```
python train.py --config configs/v14.yaml
```
每個 Epoch 結束後，程式會使用 val split 計算 EER，並將最佳模型存到 output.dir。
若 musan_path 或 rir_path 不為 null，訓練時會啟用資料增強。若兩者皆為 null，則不使用 MUSAN/RIR 增強。

## 6. 模型評估
若要針對特定測試集或已訓練好的模型進行 EER 評估，請執行：

```bash
python eval.py --config configs/v14.yaml
```
評估會根據 config 中的設定抽取語者與語音數量，並計算 EER。


