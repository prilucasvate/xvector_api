import Foundation
import OnnxRuntimeBindings

/// 與 api/app.py 的 /compare 相同欄位。snormScore 只有在提供 cohort 時才有值。
public struct Score: Equatable {
    public let rawScore: Float
    public let snormScore: Float?
    public let threshold: Float
    public let isSamePerson: Bool
}

/// X-Vector 語者特徵提取與比對 (ONNX Runtime)。
public final class SpeakerEncoder {
    /// 與 api/app.py 相同的判定門檻
    public static let threshold: Float = 0.17
    public static let snormThreshold: Float = 9.5

    private let env: ORTEnv
    private let session: ORTSession
    private let cohort: [Float]?

    public var hasCohort: Bool { cohort != nil }

    /// - Parameters:
    ///   - modelURL: mobile/export_onnx.py 輸出的 ONNX 模型
    ///   - cohortURL: mobile/export_onnx.py 輸出的 cohort.bin，提供時以 S-Norm 判定
    public init(modelURL: URL, cohortURL: URL? = nil) throws {
        env = try ORTEnv(loggingLevel: .warning)
        session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: nil)
        cohort = try cohortURL.map { url in
            let data = try Data(contentsOf: url)
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
    }

    /// - Parameter waveform: 已經過 `Audio.preprocess` 的 16kHz 單聲道
    public func embed(_ waveform: [Float]) throws -> [Float] {
        let data = waveform.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * 4) }
        let input = try ORTValue(tensorData: data, elementType: .float, shape: [1, NSNumber(value: waveform.count)])
        let outputs = try session.run(withInputs: ["waveform": input], outputNames: Set(["embedding"]), runOptions: nil)
        guard let output = outputs["embedding"] else { throw XVectorError.inference("模型沒有輸出 embedding") }
        let out = try output.tensorData() as Data
        return out.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// 直接從音檔取得 embedding。耗時操作，請勿在主執行緒呼叫。
    public func embed(url: URL) throws -> [Float] {
        try embed(Audio.preprocess(Audio.decode(url: url)))
    }

    /// - Parameter useSnorm: 有 cohort 時預設使用 S-Norm，與 API 相同
    public func score(_ e1: [Float], _ e2: [Float], useSnorm: Bool = true) -> Score {
        let raw = Self.dot(e1, e2) // embedding 已 L2 正規化
        guard let c = cohort, useSnorm else {
            return Score(rawScore: raw, snormScore: nil, threshold: Self.threshold, isSamePerson: raw > Self.threshold)
        }
        let (m1, s1) = Self.cohortStats(c, e1)
        let (m2, s2) = Self.cohortStats(c, e2)
        let snorm = ((raw - m1) / (s1 + 1e-6) + (raw - m2) / (s2 + 1e-6)) / 2
        return Score(rawScore: raw, snormScore: snorm, threshold: Self.snormThreshold, isSamePerson: snorm > Self.snormThreshold)
    }

    public static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var d: Float = 0
        for i in a.indices { d += a[i] * b[i] }
        return d
    }

    private static func cohortStats(_ c: [Float], _ e: [Float]) -> (Float, Float) {
        let dim = e.count
        let n = c.count / dim
        var s = 0.0, s2 = 0.0
        for k in 0..<n {
            var d = 0.0
            for i in 0..<dim { d += Double(c[k * dim + i] * e[i]) }
            s += d
            s2 += d * d
        }
        let mean = s / Double(n)
        return (Float(mean), Float(sqrt(max(s2 / Double(n) - mean * mean, 0))))
    }
}
