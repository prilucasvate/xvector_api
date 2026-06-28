# Configs

此資料夾存放訓練與評估設定檔。

主要設定檔：
```v14.yaml```：V14 訓練與評估設定檔。
欄位說明
```yaml
data:
  manifest: data/sample_manifest.jsonl
  data_root: data/wavs
```
* manifest：JSONL manifest 路徑。
* data_root：相對音檔路徑的根目錄。若 manifest 中的 path 已是絕對路徑，可設為 null。
```yaml
output:
  dir: outputs/v14
```
* dir：訓練輸出資料夾，包含 checkpoint 與 log。
```yaml
training:
  seed: 42
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  num_workers: 4
  weight_decay: 0.0001
```
* 訓練超參數。
```yaml
augmentation:
  musan_path: null
  rir_path: null
```
* musan_path：MUSAN noise 資料夾路徑。
* rir_path：預處理後的 RIR cache 路徑。
若兩者皆為 null，則不使用 MUSAN/RIR 資料增強。
```yaml
evaluation:
  checkpoint: outputs/v14/best_model_plus_v14.pth
  split: test
  num_spks: 50
  utts_per_spk: 10
  rounds: 10
```
* checkpoint：要評估的模型權重。
* split：評估使用的 manifest split，通常為 test。
* num_spks：每輪抽取的語者數。
* utts_per_spk：每位語者抽取的語音數。
* rounds：重複評估輪數。