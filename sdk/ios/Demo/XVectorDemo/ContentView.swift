import AVFoundation
import SwiftUI
import UniformTypeIdentifiers
import XVector

struct Slot {
    var name = "選擇音檔或直接錄音"
    var meta = ""
    var embedding: [Float]?
}

@MainActor
final class DemoModel: ObservableObject {
    @Published var slots = [Slot(), Slot()]
    @Published var status = "模型載入中…"
    @Published var useSnorm = true
    @Published var recordingSlot: Int?
    @Published var recordSeconds = 0
    @Published var ready = false

    private var encoder: SpeakerEncoder?
    private var recorder: Recorder?
    private var timer: Timer?
    private let queue = DispatchQueue(label: "xvector")

    var score: Score? {
        guard let e1 = slots[0].embedding, let e2 = slots[1].embedding, let encoder else { return nil }
        return encoder.score(e1, e2, useSnorm: useSnorm)
    }

    func load() {
        queue.async {
            do {
                let enc = try SpeakerEncoder(
                    modelURL: Bundle.main.url(forResource: "xvector_v14", withExtension: "onnx")!,
                    cohortURL: Bundle.main.url(forResource: "cohort", withExtension: "bin"))
                Task { @MainActor in
                    self.encoder = enc
                    self.ready = true
                    self.status = "模型已載入，請選擇或錄製兩段音訊。"
                }
            } catch {
                Task { @MainActor in self.status = "模型載入失敗：\(error.localizedDescription)" }
            }
        }
    }

    func analyzeFile(_ i: Int, url: URL) {
        analyze(i, name: url.lastPathComponent) {
            // fileImporter 取得的檔案需要取得存取權
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            return Audio.preprocess(try Audio.decode(url: url))
        }
    }

    /// 解碼與推論在背景執行，完成後更新畫面
    private func analyze(_ i: Int, name: String, load: @escaping () throws -> [Float]) {
        guard let encoder else { return }
        slots[i] = Slot(name: name, meta: "分析中…")
        queue.async {
            do {
                let wav = try load()
                let t0 = Date()
                let e = try encoder.embed(wav)
                let ms = Int(Date().timeIntervalSince(t0) * 1000)
                Task { @MainActor in
                    self.slots[i].embedding = e
                    self.slots[i].meta = String(format: "%.1f 秒・推論 %d ms", Double(wav.count) / Double(Audio.sampleRate), ms)
                    self.status = ""
                }
            } catch {
                Task { @MainActor in
                    self.slots[i].meta = ""
                    self.status = "無法處理 \(name)：\(error.localizedDescription)"
                }
            }
        }
    }

    func toggleRecording(_ i: Int) {
        if let r = recorder {
            guard recordingSlot == i else { return }
            timer?.invalidate()
            recorder = nil
            recordingSlot = nil
            let samples = r.stop()
            if samples.count < Audio.sampleRate / 2 {
                status = "錄音太短，請再試一次"
                return
            }
            analyze(i, name: "錄音 \(i == 0 ? "A" : "B")") { Audio.preprocess([samples]) }
            return
        }
        AVAudioSession.sharedInstance().requestRecordPermission { granted in
            Task { @MainActor in
                guard granted else {
                    self.status = "需要麥克風權限才能錄音，請到「設定」開啟"
                    return
                }
                self.startRecording(i)
            }
        }
    }

    private func startRecording(_ i: Int) {
        let r = Recorder()
        do {
            try r.start()
        } catch {
            status = "無法開始錄音：\(error.localizedDescription)"
            return
        }
        recorder = r
        recordingSlot = i
        recordSeconds = 0
        status = "錄音中，建議至少說 3 秒…"
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.recordSeconds = Int(r.seconds) }
        }
    }
}

struct ContentView: View {
    @StateObject private var model = DemoModel()
    @State private var importing: Int?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("X-Vector 語者比對").font(.title2.bold())
                Text("選擇或錄製兩段音訊，在手機上離線判斷是否為同一人。")
                    .foregroundStyle(.secondary)

                ForEach(0..<2, id: \.self) { i in slotCard(i) }

                Toggle("使用 S-Norm (與 API 預設相同)", isOn: $model.useSnorm)

                if let s = model.score {
                    resultCard(s)
                }
                Text(model.status).foregroundStyle(.secondary)
            }
            .padding()
        }
        .fileImporter(
            isPresented: Binding(get: { importing != nil }, set: { if !$0 { importing = nil } }),
            allowedContentTypes: [.audio]
        ) { result in
            if let i = importing, case .success(let url) = result { model.analyzeFile(i, url: url) }
            importing = nil
        }
        .onAppear { model.load() }
    }

    private func slotCard(_ i: Int) -> some View {
        let slot = model.slots[i]
        let recordingHere = model.recordingSlot == i
        return VStack(alignment: .leading, spacing: 6) {
            Text("音檔 \(i == 0 ? "A" : "B")").font(.caption.bold()).foregroundStyle(.secondary)
            Text(slot.name).font(.headline)
            if !slot.meta.isEmpty {
                Text(slot.meta).font(.footnote).foregroundStyle(.secondary)
            }
            HStack {
                Button("選擇檔案") { importing = i }
                    .buttonStyle(.bordered)
                    .disabled(!model.ready || model.recordingSlot != nil)
                Button(recordingHere ? "■ 停止 \(model.recordSeconds)s" : (slot.name.hasPrefix("錄音") ? "● 重新錄音" : "● 錄音")) {
                    model.toggleRecording(i)
                }
                .buttonStyle(.bordered)
                .tint(recordingHere ? .red : .accentColor)
                .disabled(!model.ready || (model.recordingSlot != nil && !recordingHere))
            }
            .padding(.top, 4)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.secondary.opacity(0.3)))
    }

    private func resultCard(_ s: Score) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(s.isSamePerson ? "判定：同一人" : "判定：不同人")
                .font(.title2.bold())
                .foregroundStyle(s.isSamePerson ? .green : .orange)
            HStack {
                number("Cosine", String(format: "%.4f", s.rawScore))
                number("S-Norm", s.snormScore.map { String(format: "%.2f", $0) } ?? "–")
                number("門檻", String(s.threshold))
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
    }

    private func number(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.monospacedDigit())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
