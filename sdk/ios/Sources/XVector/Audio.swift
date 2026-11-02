import AVFoundation
import Foundation

/// 音訊前處理，對齊 api/inference.py：
/// 各聲道重新取樣到 16kHz -> 多聲道取平均 -> RMS 標準化到 0.05 -> clamp 到 [-1, 1]。
public enum Audio {
    public static let sampleRate = 16000
    public static let targetRMS: Float = 0.05

    /// 多聲道取平均、RMS 標準化並 clamp。channels 需已是 16kHz。
    public static func preprocess(_ channels: [[Float]]) -> [Float] {
        let n = channels[0].count
        var out = [Float](repeating: 0, count: n)
        for ch in channels {
            for i in 0..<n { out[i] += ch[i] / Float(channels.count) }
        }
        var sum = 0.0
        for v in out { sum += Double(v * v) }
        let rms = Float(sqrt(sum / Double(n)))
        let gain = rms > 0 ? targetRMS / rms : 1
        for i in 0..<n { out[i] = min(1, max(-1, out[i] * gain)) }
        return out
    }

    /// 移植自 torchaudio.functional.resample (sinc_interp_hann,
    /// lowpass_filter_width=6, rolloff=0.99)，結果與 Python 端一致。
    public static func resample(_ x: [Float], from: Int, to: Int = sampleRate) -> [Float] {
        if from == to { return x }
        let g = gcd(from, to)
        let orig = from / g
        let new = to / g
        let lowpassWidth = 6.0
        let baseFreq = Double(min(orig, new)) * 0.99
        let width = Int(ceil(lowpassWidth * Double(orig) / baseFreq))
        let k = 2 * width + orig

        // kernel: [new][k]，torchaudio 以 float64 計算後轉為 float32
        let kernel: [[Float]] = (0..<new).map { j in
            (0..<k).map { i in
                var t = (-Double(j) / Double(new) + Double(i - width) / Double(orig)) * baseFreq
                t = min(lowpassWidth, max(-lowpassWidth, t))
                let w = pow(cos(t * .pi / lowpassWidth / 2), 2)
                let tp = t * .pi
                let s = tp == 0 ? 1.0 : sin(tp) / tp
                return Float(s * w * baseFreq / Double(orig))
            }
        }

        // 左補 width、右補 width + orig 個 0，再以 stride=orig 做 conv1d
        var padded = [Float](repeating: 0, count: x.count + 2 * width + orig)
        padded.replaceSubrange(width..<(width + x.count), with: x)
        let frames = (padded.count - k) / orig + 1
        let target = Int(ceil(Double(new) * Double(x.count) / Double(orig)))
        var out = [Float](repeating: 0, count: min(target, frames * new))
        padded.withUnsafeBufferPointer { p in
            for f in 0..<frames {
                let base = f * orig
                for j in 0..<new {
                    let idx = f * new + j
                    if idx >= out.count { return }
                    var acc = 0.0
                    let kj = kernel[j]
                    for i in 0..<k { acc += Double(kj[i]) * Double(p[base + i]) }
                    out[idx] = Float(acc)
                }
            }
        }
        return out
    }

    /// 解碼 AVFoundation 支援的音檔 (wav/mp3/m4a/aac/flac/caf...)，回傳 16kHz 的各聲道資料。
    public static func decode(url: URL) throws -> [[Float]] {
        let file = try AVAudioFile(forReading: url)
        // processingFormat 為 float32 非交錯格式，int16 會換算為 /32768，與 soundfile 相同
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw XVectorError.decode("無法配置音訊緩衝區")
        }
        try file.read(into: buffer)
        guard let data = buffer.floatChannelData else { throw XVectorError.decode("不支援的音訊格式") }
        let n = Int(buffer.frameLength)
        return (0..<Int(format.channelCount)).map { c in
            resample(Array(UnsafeBufferPointer(start: data[c], count: n)), from: Int(format.sampleRate))
        }
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
}

public enum XVectorError: Error, LocalizedError {
    case decode(String)
    case inference(String)

    public var errorDescription: String? {
        switch self {
        case .decode(let m), .inference(let m): return m
        }
    }
}
