// 以 Node 執行 onnxruntime-web (WASM)，比對 PyTorch 產生的 golden.json
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { SpeakerEncoder, preprocess, cosine } from '../src/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const mobile = join(here, '..', '..', '..', 'mobile');

// 只解析測試用的 float32 單聲道 wav
function readWav(path) {
  const buf = readFileSync(path);
  let off = 12;
  while (off < buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    if (id === 'data') {
      const bytes = buf.subarray(off + 8, off + 8 + size);
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
    }
    off += 8 + size;
  }
  throw new Error('no data chunk');
}

const golden = JSON.parse(readFileSync(join(here, 'golden.json'), 'utf8'));
const cohort = new Float32Array(readFileSync(join(mobile, 'cohort.bin')).buffer.slice(0));
const encoder = await SpeakerEncoder.create(new Uint8Array(readFileSync(join(mobile, 'xvector_v14.onnx'))), { cohort });

let ok = true;
const emb = {};
for (const name of ['a', 'b']) {
  emb[name] = await encoder.embed(preprocess([readWav(join(here, `${name}.wav`))]));
  const c = cosine(emb[name], golden[name]);
  console.log(`[Test] embedding ${name}: cosine vs PyTorch = ${c.toFixed(6)}`);
  ok &&= c > 0.9999;
}
const s = encoder.score(emb.a, emb.b);
console.log('[Test] JS     :', s);
console.log('[Test] PyTorch:', golden.score);
ok &&= Math.abs(s.raw_score - golden.score.raw_score) < 1e-3 && Math.abs(s.snorm_score - golden.score.snorm_score) < 1e-2;
console.log(ok ? '[Test] PASS' : '[Test] FAIL');
process.exit(ok ? 0 : 1);
