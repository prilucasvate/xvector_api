# XVector (iOS)

在 iOS 以 ONNX Runtime 離線執行 X-Vector 語者比對，前處理與分數計算對齊 `api/` 的 FastAPI 服務，
介面與 `sdk/android` 相同。

> ⚠️ 此版本尚未在 Mac / iPhone 上編譯與測試，請先執行下方的測試確認。

## 安裝 (Swift Package Manager)
Xcode → File → Add Package Dependencies，選擇本 repo 的 `sdk/ios` 目錄，加入 `XVector`。

## 使用方式
```swift
import XVector

let encoder = try SpeakerEncoder(
    modelURL: Bundle.main.url(forResource: "xvector_v14", withExtension: "onnx")!,
    cohortURL: Bundle.main.url(forResource: "cohort", withExtension: "bin"))
// 解碼與推論較耗時，請在背景執行
let e1 = try encoder.embed(url: fileA)   // wav / mp3 / m4a / aac / flac / caf
let e2 = try encoder.embed(url: fileB)
let score = encoder.score(e1, e2)        // Score(rawScore, snormScore, threshold, isSamePerson)
```
自行錄音時，可用 `Audio.resample(samples, from: sampleRate)` 轉為 16kHz，
再呼叫 `encoder.embed(Audio.preprocess([samples]))`。

## 測試 (macOS)
```bash
python mobile/export_onnx.py                  # 產生 xvector_v14.onnx 與 cohort.bin
python sdk/android/tools/make_golden.py       # 用 PyTorch 產生參考結果
cd sdk/ios && swift test
```

## Demo App
```bash
brew install xcodegen
cd sdk/ios/Demo && xcodegen && open XVectorDemo.xcodeproj
```
在 Xcode 設定 Signing Team 後即可安裝到 iPhone。可選擇檔案或用麥克風錄音，比對兩段音訊是否為同一人。
