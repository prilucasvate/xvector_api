# xvector-android

在 Android 以 ONNX Runtime 離線執行 X-Vector 語者比對，前處理與分數計算對齊 `api/` 的 FastAPI 服務。

* `xvector/`：Kotlin library (AAR)。以 MediaCodec 解碼 wav / mp3 / m4a / ogg / flac，重新取樣移植自 torchaudio，結果與 Python 一致。
* `demo/`：Demo App，可選擇音檔或用麥克風錄音，比對兩段音訊是否為同一人。

## 建置
```bash
cd mobile && python export_onnx.py        # 產生 xvector_v14.onnx 與 cohort.bin (會自動打包進 Demo App)
cd ../sdk/android
./gradlew :demo:assembleRelease           # demo/build/outputs/apk/release/demo-arm64-v8a-release-unsigned.apk
./gradlew :xvector:assembleRelease        # xvector/build/outputs/aar/xvector-release.aar
```

## 使用方式
```kotlin
val encoder = SpeakerEncoder.fromAssets(context, "xvector_v14.onnx", "cohort.bin")
// 解碼與推論較耗時，請在背景執行緒呼叫
val e1 = encoder.embed(context, uriA)
val e2 = encoder.embed(context, uriB)
val score = encoder.score(e1, e2)   // Score(rawScore, snormScore, threshold, isSamePerson)
```
自行錄音時可直接以 16kHz 單聲道 float 取得資料，再呼叫 `encoder.embed(Audio.preprocess(listOf(samples)))`。

## 測試
```bash
python tools/make_golden.py                # 用 PyTorch 產生參考結果 (含 44.1kHz 雙聲道音檔)
./gradlew :xvector:connectedDebugAndroidTest
```

## 驗證結果 (Android 15 模擬器，x86_64)
| 項目 | 結果 |
|---|---|
| 44.1kHz 雙聲道 wav，與 PyTorch 的 Cosine | 0.9999999 |
| 16kHz 單聲道 wav，與 PyTorch 的 Cosine | 0.9999996 |
| Raw / S-Norm 分數 | 0.21545 / 5.1880 (PyTorch：0.21545 / 5.1880) |
| 3 秒音訊推論時間 | 約 70–80 ms |

## 注意事項
* Demo APK 依 CPU 架構拆分：`arm64-v8a` 給實體手機、`x86_64` 給模擬器。
  release 版啟用 R8 並壓縮 `libonnxruntime.so`，arm64 APK 約 38 MB (原本 109 MB)，其中模型佔 28 MB。
  若需要更小，可改用 int8 模型 (7 MB)，但對 EER 的影響尚未評估。
* App 使用 R8 時需保留 ONNX Runtime 的 JNI 類別，library 已透過 `consumer-rules.pro` 自動帶入。
* 錄音使用 `VOICE_RECOGNITION` 音源，關閉大部分降噪與自動增益，較接近訓練資料。
