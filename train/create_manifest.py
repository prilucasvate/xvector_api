import os
import json
import random
import soundfile as sf

# ================= 設定區 =================
DATA_DIR = "data/wavs"               # 音檔總目錄 (裡面要是語者名稱的子資料夾)
OUTPUT_JSONL = "data/manifest.jsonl" # 輸出的清單路徑

MIN_DURATION_SEC = 2.0  # 最短音檔限制：2 秒
MIN_UTTERANCES = 15     # 每個語者最少需要的有效音檔數 (不夠的人直接刪除)
OPEN_SET_RATIO = 0.1    # 10% 的語者作為完全未知的測試集 (Test_Open)
TRAIN_RATIO = 0.9       # 剩餘語者中，90% 拿來 Train，10% 拿來 Test_Closed
SEED = 42               # 亂數種子，確保每次切分結果一致
# =========================================

def create_manifest():
    if not os.path.exists(DATA_DIR):
        print(f"[Create] can't find directory: {DATA_DIR} . Please check the path and try again.")
        return

    spk2data = {}
    stats_short_audio = 0
    broken_files = 0

    print(f"[Create] Starting recursive scan and inspection of audio files...")
    
    # 遍歷資料夾 (data/wavs/speaker_01/xxx.wav)
    for spk_id in os.listdir(DATA_DIR):
        spk_dir = os.path.join(DATA_DIR, spk_id)
        if not os.path.isdir(spk_dir):
            continue
            
        spk2data[spk_id] = []
        for file in os.listdir(spk_dir):
            if file.endswith(('.wav', '.flac')):
                filepath = os.path.join(spk_dir, file)
                
                # --- check ---
                try:
                    info = sf.info(filepath)
                    duration = info.duration  # soundfile 直接提供秒數
                    
                    if duration < MIN_DURATION_SEC:
                        stats_short_audio += 1
                        continue # 太短，丟掉！
                        
                    spk2data[spk_id].append({
                        "path": filepath,
                        "spk_id": spk_id,
                        "duration": round(duration, 4)
                    })
                except Exception as e:
                    broken_files += 1
                    print(f"[Create] Error occurred while processing {filepath}: {e}")
                # ---  ---

    print("\n=== Filtering and Splitting ===")
    valid_spks = {}
    stats_skipped_spks = 0
    
    # 每個語者至少要有 MIN_UTTERANCES 句話
    for spk_id, items in spk2data.items():
        if len(items) >= MIN_UTTERANCES:
            valid_spks[spk_id] = items
        else:
            stats_skipped_spks += 1

    if not valid_spks:
        print(f"[Create] No speakers meet the requirement of at least {MIN_UTTERANCES} utterances! Please check your dataset and try again.")
        return

    # 準備切分 Open-set 與 Closed-set
    all_valid_spk_ids = sorted(list(valid_spks.keys()))
    random.seed(SEED)
    random.shuffle(all_valid_spk_ids)

    # 抽出 10% 的人當作完全陌生的 Open-set
    num_open_spks = max(1, int(len(all_valid_spk_ids) * OPEN_SET_RATIO))
    open_set_spks = set(all_valid_spk_ids[:num_open_spks])

    final_data = []
    for spk_id, items in valid_spks.items():
        if spk_id in open_set_spks:
            # 這些人不參與訓練，全部標記為 test_open
            for it in items:
                it['split'] = 'test_open'
                final_data.append(it)
        else:
            # 參與訓練的人 (Closed-set)
            random.shuffle(items)
            split_idx = int(len(items) * TRAIN_RATIO)
            # 確保至少有一句話拿來做 test_closed 算 EER
            if len(items) - split_idx < 1:
                split_idx = len(items) - 1

            for it in items[:split_idx]:
                it['split'] = 'train'
                final_data.append(it)
            for it in items[split_idx:]:
                it['split'] = 'test_closed'
                final_data.append(it)

    # 寫入 JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 統計數據
    train_count = sum(1 for x in final_data if x['split'] == 'train')
    test_closed_count = sum(1 for x in final_data if x['split'] == 'test_closed')
    test_open_count = sum(1 for x in final_data if x['split'] == 'test_open')

    print("\n" + "="*50)
    print("      X-VECTOR DATASET REPORT      ")
    print("="*50)
    print(f"Too short (<{MIN_DURATION_SEC}s) removed audio files: {stats_short_audio} files")
    print(f"Broken audio files:           {broken_files} files")
    print(f"Speakers skipped (insufficient utterances): {stats_skipped_spks} persons")
    print("-" * 50)
    print(f"Total Speakers: {len(valid_spks)} persons")
    print(f"  ├─ Closed-set : {len(valid_spks) - len(open_set_spks)} persons")
    print(f"  └─ Open-set :   {len(open_set_spks)} persons")
    print("-" * 50)
    print(f"Dataset Distribution:")
    print(f"  ├─ Train :       {train_count}")
    print(f"  ├─ Test_closed : {test_closed_count}")
    print(f"  └─ Test_open :   {test_open_count}")
    print("="*50)

if __name__ == "__main__":
    create_manifest()