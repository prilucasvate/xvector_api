# X-Vector Speaker Embedding Service
## 簡介
這程式結合了從原始音訊前處理、升級 X-Vector 模型訓練，到 FastAPI 推論介面，支援 Zero-shot 語者嵌入向量（Speaker Embedding）提取和比對。  
透過 TDNN 結構與注意力機制，模型能夠將語音訊號壓縮為固定的 512 維聲紋向量。這些向量可用於語者驗證、語者辨識等任務。
## 功能
* /extract: 上傳一個 wav 音檔，回傳對應的 512 維 X-Vector 特徵向量。
* /compare: 上傳兩個 wav 音檔，回傳它們的 Cosine Similarity 相似度，以及是否判定為同一人。
## 模型介紹
* 語音前處理 (Mel-Spec): 使用 80 維 Mel-Spectrogram 加上分貝縮放。
* 特徵提取 (TDNN + SE): 強化 TDNN ，加上 SE Block (Squeeze-and-Excitation) **通道注意力**加權。
* 重點提取: 使用 Attentive Statistics Pooling (ASP)，透過**時間注意力**機制動態計算平均值和標準差。
* Loss Function: 使用 ArcFace (AAM-Softmax)，在 512 維上最大化間距。
* Performance: 在語者資料集測試中達到約 5.48% EER。
## 目錄
.
├── api/                    # API 相關  
│   ├── app.py              # FastAPI 主程式  
│   ├── inference.py        # 推論核心  
│   └── weights/            # 存放訓練好的模型 (.pth)  
├── train/                  # 訓練相關  
│   ├── data/               # 存放 manifest.jsonl 與 wav 原始檔  
│   ├── create_manifest.py  # 資料集檢驗和切分  
│   ├── dataset.py          # DataLoader 和資料增強 (MUSAN)  
│   ├── model.py            # X-Vector 主模型  
│   └── train.py            # 主訓練腳本 (EER 驗證)  
├── Dockerfile              # 容器部署設定  
└── requirements.txt        # 環境清單  

## 使用說明
1. 環境準備
建議使用 Python 3.11+ 以及 CUDA 支援的環境：
`pip install -r requirements.txt`
2. 資料集準備與體檢
將音檔放入 train/data/wavs/<語者ID>/<音檔>.wav，之後執行：
```
cd train
python create_manifest.py
```
這會自動檢查音檔長度（> 2s）與檔案完整性，並按 9:1 自動分訓練與驗證集。

3. 開始訓練
執行主訓練腳本，可透過環境變數調整 CPU Worker 數量：
`NUM_WORKERS=8 python train.py`
4. 啟動 API 服務
將訓練好的 best_model.pth 放入 api/weights/，執行：
```
cd api
python app.py
```
啟動後，開啟 http://localhost:8000/docs 即可進入測試介面。