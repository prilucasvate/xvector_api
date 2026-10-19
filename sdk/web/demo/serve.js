// Demo 用的靜態伺服器：node demo/serve.js [port]
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join, normalize, extname } from 'node:path';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const routes = {
  '/ort/': join(web, 'node_modules', 'onnxruntime-web', 'dist'),
  '/src/': join(web, 'src'),
  '/models/': join(web, '..', '..', 'mobile'),
  '/': join(web, 'demo'),
};
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm' };
const port = Number(process.argv[2] ?? 8080);

createServer(async (req, res) => {
  let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (path === '/') path = '/index.html';
  const prefix = Object.keys(routes).find((p) => path.startsWith(p));
  const base = routes[prefix];
  const file = normalize(join(base, path.slice(prefix.length)));
  if (!file.startsWith(base)) return res.writeHead(403).end();
  try {
    const body = await readFile(file);
    // COOP/COEP 讓 WASM 可以使用多執行緒
    res.writeHead(200, {
      'Content-Type': types[extname(file)] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
}).listen(port, () => console.log(`[Demo] http://localhost:${port}`));
