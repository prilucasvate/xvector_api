package tw.xvector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.net.Uri
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.sqrt

/** 與 api/app.py 的 /compare 相同欄位。snormScore 只有在提供 cohort 時才有值。 */
data class Score(
    val rawScore: Float,
    val snormScore: Float?,
    val threshold: Float,
    val isSamePerson: Boolean,
)

/**
 * X-Vector 語者特徵提取與比對 (ONNX Runtime Android)。
 *
 * @param model mobile/export_onnx.py 輸出的 ONNX 模型位元組
 * @param cohort mobile/export_onnx.py 輸出的 cohort.bin 位元組，提供時以 S-Norm 判定
 */
class SpeakerEncoder(model: ByteArray, cohort: ByteArray? = null) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(model, OrtSession.SessionOptions())
    private val cohort: FloatArray? = cohort?.let {
        val fb = ByteBuffer.wrap(it).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        FloatArray(fb.remaining()).also { arr -> fb.get(arr) }
    }

    val hasCohort get() = cohort != null

    /** @param waveform 已經過 [Audio.preprocess] 的 16kHz 單聲道 */
    fun embed(waveform: FloatArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(waveform), longArrayOf(1, waveform.size.toLong())).use { input ->
            session.run(mapOf("waveform" to input)).use { result ->
                @Suppress("UNCHECKED_CAST")
                return (result[0].value as Array<FloatArray>)[0]
            }
        }
    }

    /** 直接從音檔 Uri 取得 embedding。耗時操作，請勿在主執行緒呼叫。 */
    fun embed(context: Context, uri: Uri): FloatArray = embed(Audio.preprocess(Audio.decode(context, uri)))

    /** @param useSnorm 有 cohort 時預設使用 S-Norm，與 API 相同 */
    fun score(e1: FloatArray, e2: FloatArray, useSnorm: Boolean = true): Score {
        val raw = dot(e1, e2) // embedding 已 L2 正規化
        val c = cohort
        if (c == null || !useSnorm) {
            return Score(raw, null, THRESHOLD, raw > THRESHOLD)
        }
        val (m1, s1) = cohortStats(c, e1)
        val (m2, s2) = cohortStats(c, e2)
        val snorm = ((raw - m1) / (s1 + 1e-6f) + (raw - m2) / (s2 + 1e-6f)) / 2f
        return Score(raw, snorm, SNORM_THRESHOLD, snorm > SNORM_THRESHOLD)
    }

    private fun cohortStats(c: FloatArray, e: FloatArray): Pair<Float, Float> {
        val dim = e.size
        val n = c.size / dim
        var s = 0.0
        var s2 = 0.0
        for (k in 0 until n) {
            var d = 0.0
            for (i in 0 until dim) d += c[k * dim + i] * e[i]
            s += d
            s2 += d * d
        }
        val mean = s / n
        return mean.toFloat() to sqrt(maxOf(s2 / n - mean * mean, 0.0)).toFloat()
    }

    override fun close() = session.close()

    companion object {
        // 與 api/app.py 相同的判定門檻
        const val THRESHOLD = 0.17f
        const val SNORM_THRESHOLD = 9.5f

        fun dot(a: FloatArray, b: FloatArray): Float {
            var d = 0f
            for (i in a.indices) d += a[i] * b[i]
            return d
        }

        /** 從 App 的 assets 載入模型與 cohort。 */
        fun fromAssets(context: Context, model: String, cohort: String? = null) = SpeakerEncoder(
            context.assets.open(model).use { it.readBytes() },
            cohort?.let { name -> context.assets.open(name).use { it.readBytes() } },
        )
    }
}
