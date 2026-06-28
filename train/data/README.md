此資料夾用於放置 manifest 與音檔資料。

## Manifest

`sample_manifest.jsonl` 是公開範例檔，用於展示 JSONL 欄位格式。實際訓練時可在 `configs/v14.yaml` 中指定其他路徑。

必要欄位：
```json
{"path":"speaker_01/utt_001.wav","spk_id":"speaker_01","split":"train","duration":3.2}
```
欄位說明：
* path：音檔路徑，可為絕對路徑，或相對於 config 中的 data_root。
* spk_id：語者 ID。
* split：資料切分，使用 train、val、test。
* duration：音檔秒數。  

## Local Audio Layout
若使用 create_manifest.py 產生自備資料集 manifest，請將音檔放成：
```text
data/wavs/
├── speaker_01/
│   ├── utt_001.wav
│   └── utt_002.wav
└── speaker_02/  
    ├── utt_001.wav
    └── utt_002.wav
```