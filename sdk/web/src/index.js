/**
 * xvector-web：在瀏覽器以 ONNX Runtime Web (WASM) 離線執行 X-Vector 語者比對。
 * 前處理與分數計算對齊 api/inference.py 與 api/app.py。
 */
import * as ort from 'onnxruntime-web';

export const SAMPLE_RATE = 16000;
export const TARGET_RMS = 0.05;
// 與 api/app.py 相同的判定門檻
export const THRESHOLD = 0.17;
export const SNORM_THRESHOLD = 9.5;

/**
 * 多聲道取平均、RMS 標準化到 0.05 並 clamp 到 [-1, 1]。
 * @param {Float32Array[]} channels 已是 16kHz 的各聲道資料
 */
export function preprocess(channels) {
  const n = channels[0].length;
  const out = new Float32Array(n);
  for (const ch of channels) for (let i = 0; i < n; i++) out[i] += ch[i] / channels.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += out[i] * out[i];
  const rms = Math.sqrt(sum / n);
  const gain = rms > 0 ? TARGET_RMS / rms : 1;
  for (let i = 0; i < n; i++) out[i] = Math.min(1, Math.max(-1, out[i] * gain));
  return out;
}

/**
 * 瀏覽器專用：將任意格式音檔 (wav/mp3/m4a/ogg...) 解碼並重新取樣為 16kHz 各聲道資料。
 * @param {ArrayBuffer|Blob} input
 */
export async function decodeAudio(input) {
  const buf = input instanceof ArrayBuffer ? input : await input.arrayBuffer();
  // decodeAudioData 會直接重新取樣到 context 的取樣率；
  // 若用一般 AudioContext (通常 48kHz) 再轉 16kHz 會重複取樣，結果與 Python 不一致
  const resampled = await new OfflineAudioContext(1, 1, SAMPLE_RATE).decodeAudioData(buf);
  const channels = [];
  for (let c = 0; c < resampled.numberOfChannels; c++) channels.push(resampled.getChannelData(c));
  return channels;
}

export function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // embedding 已 L2 正規化
}

export class SpeakerEncoder {
  /**
   * @param {ort.InferenceSession} session
   * @param {Float32Array|null} cohort 已 L2 正規化的 cohort，[N * 512]
   */
  constructor(session, cohort = null) {
    this.session = session;
    this.cohort = cohort;
  }

  /**
   * @param {string|Uint8Array} model ONNX 模型的 URL 或位元組
   * @param {{cohort?: string|Float32Array, wasmPaths?: string}} options
   */
  static async create(model, options = {}) {
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
    const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
    let cohort = options.cohort ?? null;
    if (typeof cohort === 'string') cohort = new Float32Array(await (await fetch(cohort)).arrayBuffer());
    return new SpeakerEncoder(session, cohort);
  }

  /** @param {Float32Array} waveform 已前處理的 16kHz 單聲道 */
  async embed(waveform) {
    const input = new ort.Tensor('float32', waveform, [1, waveform.length]);
    const { embedding } = await this.session.run({ waveform: input });
    return embedding.data;
  }

  /** 瀏覽器專用：直接從音檔取得 embedding */
  async embedFile(file) {
    return this.embed(preprocess(await decodeAudio(file)));
  }

  /** 回傳與 /compare API 相同欄位的結果 */
  score(e1, e2) {
    const raw = cosine(e1, e2);
    if (!this.cohort) {
      return { raw_score: raw, snorm_score: null, threshold: THRESHOLD, is_same_person: raw > THRESHOLD };
    }
    const dim = e1.length;
    const stats = (e) => {
      const n = this.cohort.length / dim;
      let s = 0, s2 = 0;
      for (let k = 0; k < n; k++) {
        let d = 0;
        for (let i = 0; i < dim; i++) d += this.cohort[k * dim + i] * e[i];
        s += d; s2 += d * d;
      }
      const mean = s / n;
      return [mean, Math.sqrt(Math.max(s2 / n - mean * mean, 0))];
    };
    const [m1, s1] = stats(e1);
    const [m2, s2] = stats(e2);
    const snorm = ((raw - m1) / (s1 + 1e-6) + (raw - m2) / (s2 + 1e-6)) / 2;
    return { raw_score: raw, snorm_score: snorm, threshold: SNORM_THRESHOLD, is_same_person: snorm > SNORM_THRESHOLD };
  }
}
