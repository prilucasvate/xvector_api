import XCTest
@testable import XVector

/// 與 sdk/android/tools/make_golden.py 以 PyTorch 產生的結果比對。
/// 執行前需先產生 mobile/xvector_v14.onnx、cohort.bin 與 golden 資料。
final class GoldenTests: XCTestCase {
    private let repo = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().appendingPathComponent("../../../..").standardized
    private var golden: URL { repo.appendingPathComponent("sdk/android/xvector/build/golden") }

    func testMatchesPyTorch() throws {
        let json = try JSONSerialization.jsonObject(
            with: Data(contentsOf: golden.appendingPathComponent("golden.json"))) as! [String: Any]
        let encoder = try SpeakerEncoder(
            modelURL: repo.appendingPathComponent("mobile/xvector_v14.onnx"),
            cohortURL: repo.appendingPathComponent("mobile/cohort.bin"))

        var emb: [String: [Float]] = [:]
        for name in ["c", "d"] {
            let e = try encoder.embed(url: golden.appendingPathComponent("\(name).wav"))
            let ref = (json[name] as! [Double]).map { Float($0) }
            let cos = SpeakerEncoder.dot(e, ref)
            print("[GoldenTests] \(name) cosine vs PyTorch = \(cos)")
            XCTAssertGreaterThan(cos, 0.9999)
            emb[name] = e
        }
        let s = encoder.score(emb["c"]!, emb["d"]!)
        let ref = json["score"] as! [String: Double]
        print("[GoldenTests] Swift   = \(s)")
        print("[GoldenTests] PyTorch = \(ref)")
        XCTAssertEqual(s.rawScore, Float(ref["raw_score"]!), accuracy: 1e-3)
        XCTAssertEqual(s.snormScore!, Float(ref["snorm_score"]!), accuracy: 1e-2)
    }
}
