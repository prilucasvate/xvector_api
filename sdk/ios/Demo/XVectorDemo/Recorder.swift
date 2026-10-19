import AVFoundation
import XVector

/// 以 AVAudioEngine 錄音，結束時重新取樣為 16kHz 單聲道。
final class Recorder {
    private let engine = AVAudioEngine()
    private var samples: [Float] = []
    private var sampleRate = 48000.0
    private let lock = NSLock()

    var seconds: Double {
        lock.lock(); defer { lock.unlock() }
        return Double(samples.count) / sampleRate
    }

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        // .measurement 模式會關閉大部分的降噪與自動增益，較接近訓練資料
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        sampleRate = format.sampleRate
        samples = []
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self, let data = buffer.floatChannelData else { return }
            let chunk = UnsafeBufferPointer(start: data[0], count: Int(buffer.frameLength))
            self.lock.lock()
            self.samples.append(contentsOf: chunk)
            self.lock.unlock()
        }
        try engine.start()
    }

    /// 停止錄音並回傳 16kHz 單聲道資料
    func stop() -> [Float] {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false)
        lock.lock(); defer { lock.unlock() }
        return Audio.resample(samples, from: Int(sampleRate))
    }
}
