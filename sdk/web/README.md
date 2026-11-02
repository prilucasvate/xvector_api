# xvector-web

在瀏覽器以 ONNX Runtime Web (WASM) 離線執行 X-Vector 語者比對，前處理與分數計算對齊 `api/` 的 FastAPI 服務。

## Demo
```bash
cd mobile && python export_onnx.py       # 產生 xvector_v14.onnx 與 cohort.bin
cd ../sdk/web && npm install
npm run demo                              # http://localhost:8080
```
上傳兩段任意格式音檔 (wav / mp3 / m4a / ogg…) 或直接用麥克風錄音，即可看到 Cosine Similarity、S-Norm 分數與判定結果。

## 使用方式
```js
import { SpeakerEncoder } from 'xvector-web';

const encoder = await SpeakerEncoder.create('/models/xvector_v14.onnx', {
  cohort: '/models/cohort.bin',   // 選用，提供時以 S-Norm 判定 (門檻 9.5)，否則以 Cosine 判定 (門檻 0.17)
});
const e1 = await encoder.embedFile(fileA);   // File / Blob / ArrayBuffer
const e2 = await encoder.embedFile(fileB);
encoder.score(e1, e2);
// { raw_score, snorm_score, threshold, is_same_person }
```
非瀏覽器環境 (Node、Worker) 可自行取得 16kHz 音訊後使用 `encoder.embed(preprocess([samples]))`。

## 測試
```bash
python test/make_golden.py   # 用 PyTorch 產生參考結果
npm test                     # 以 WASM 比對 embedding 與分數
```

## 注意事項
* 瀏覽器以 Web Audio 重新取樣，與 torchaudio 的演算法不同；非 16kHz 音檔的分數可能有些微差異。
* 錄音需要 secure context：`localhost` 或 HTTPS。從其他電腦連線請用 `ssh -L 8080:localhost:8080` 轉發。
* 多執行緒 WASM 需要 COOP/COEP 標頭 (`demo/serve.js` 已設定)，沒有時會自動退回單執行緒。
