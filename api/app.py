from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import os

# 匯入推論核心
from inference import SpeakerEncoder

app = FastAPI(
    title="X-Vector Speaker Embedding API",
    description="Zero-shot Speaker Embedding Extractor powered by V7 Architecture",
    version="1.0.0"
)

# ==========================================
# 參數設定與模型載入
# ==========================================
# 這裡的 NUM_CLASSES 跟訓練這顆權重時的語者總數完全一致！
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 2682)) # 預設 v7 2682，根據訓練集調整
MODEL_PATH = os.getenv("MODEL_PATH", "weights/best_model_plus_v7.pth")

print("[API] Starting API server...")
try:
    # 啟動伺服器時，把模型常駐在記憶體或顯卡裡待命
    encoder = SpeakerEncoder(model_path=MODEL_PATH, num_classes=NUM_CLASSES)
except Exception as e:
    print(f"[API] Model loading failed! Please check the path and num_classes. Error: {e}")
    encoder = None

# ==========================================
# API 
# ==========================================
@app.post("/extract")
async def extract_vector(file: UploadFile = File(...)):
    """上傳一個wav，回傳 512 維的 X-vector 特徵向量"""
    if encoder is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    
    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
    
    try:
        # 讀取檔案為記憶體 Bytes，並交給 Inference 引擎
        audio_bytes = await file.read()
        embedding = encoder.extract_embedding(audio_bytes)
        
        return {
            "filename": file.filename,
            # Numpy array 必須轉成 Python 的 List 才能 JSON 回傳
            "embedding": embedding.tolist() 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"特徵提取失敗: {str(e)}")

@app.post("/compare")
async def compare_speakers(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """上傳兩個音檔，回傳 Cosine Similarity 和 是否為同一人"""
    if encoder is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
        
    try:
        bytes1 = await file1.read()
        bytes2 = await file2.read()
        
        vec1 = encoder.extract_embedding(bytes1)
        vec2 = encoder.extract_embedding(bytes2)
        
        # 計算 Cosine Similarity (抽出時已經做過 L2 正規化，直接做內積)
        cos_sim = float(np.dot(vec1, vec2))
        
        # 設定一個信心閾值，大於這個值就判斷為同一人 (建議根據你的 EER 圖表最佳閾值來調整)
        threshold = 0.80
        
        return {
            "file1": file1.filename,
            "file2": file2.filename,
            "similarity_score": round(cos_sim, 4), # 四捨五入到小數點後四位
            "is_same_person": bool(cos_sim > threshold)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    # 預設跑在 8000 port
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)