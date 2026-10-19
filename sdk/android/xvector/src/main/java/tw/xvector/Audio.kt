package tw.xvector

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * 音訊前處理，對齊 api/inference.py：
 * 各聲道重新取樣到 16kHz -> 多聲道取平均 -> RMS 標準化到 0.05 -> clamp 到 [-1, 1]。
 */
object Audio {
    const val SAMPLE_RATE = 16000
    const val TARGET_RMS = 0.05f

    /** 多聲道取平均、RMS 標準化並 clamp。channels 需已是 16kHz。 */
    fun preprocess(channels: List<FloatArray>): FloatArray {
        val n = channels[0].size
        val out = FloatArray(n)
        for (ch in channels) for (i in 0 until n) out[i] += ch[i] / channels.size
        var sum = 0.0
        for (v in out) sum += v * v
        val rms = sqrt(sum / n).toFloat()
        val gain = if (rms > 0f) TARGET_RMS / rms else 1f
        for (i in 0 until n) out[i] = (out[i] * gain).coerceIn(-1f, 1f)
        return out
    }

    /**
     * 移植自 torchaudio.functional.resample (sinc_interp_hann,
     * lowpass_filter_width=6, rolloff=0.99)，結果與 Python 端一致。
     */
    fun resample(x: FloatArray, from: Int, to: Int = SAMPLE_RATE): FloatArray {
        if (from == to) return x
        val g = gcd(from, to)
        val orig = from / g
        val new = to / g
        val lowpassWidth = 6.0
        val baseFreq = min(orig, new) * 0.99
        val width = ceil(lowpassWidth * orig / baseFreq).toInt()
        val k = 2 * width + orig

        // kernel: [new][k]，torchaudio 以 float64 計算後轉為 float32
        val kernel = Array(new) { j ->
            FloatArray(k) { i ->
                var t = (-j.toDouble() / new + (i - width).toDouble() / orig) * baseFreq
                t = t.coerceIn(-lowpassWidth, lowpassWidth)
                val w = cos(t * PI / lowpassWidth / 2).let { it * it }
                val tp = t * PI
                val s = if (tp == 0.0) 1.0 else sin(tp) / tp
                (s * w * baseFreq / orig).toFloat()
            }
        }

        // 左補 width、右補 width + orig 個 0，再以 stride=orig 做 conv1d
        val padded = FloatArray(x.size + 2 * width + orig)
        System.arraycopy(x, 0, padded, width, x.size)
        val frames = (padded.size - k) / orig + 1
        val target = ceil(new.toDouble() * x.size / orig).toInt()
        val out = FloatArray(min(target, frames * new))
        for (f in 0 until frames) {
            val base = f * orig
            for (j in 0 until new) {
                val idx = f * new + j
                if (idx >= out.size) return out
                val kj = kernel[j]
                var acc = 0.0
                for (i in 0 until k) acc += kj[i] * padded[base + i]
                out[idx] = acc.toFloat()
            }
        }
        return out
    }

    /** 解碼任意 Android 支援的音檔 (wav/mp3/m4a/ogg/flac...)，回傳 16kHz 的各聲道資料。 */
    fun decode(context: Context, uri: Uri): List<FloatArray> {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        try {
            return decode(extractor)
        } finally {
            extractor.release()
        }
    }

    /** 從檔案路徑解碼。 */
    fun decode(path: String): List<FloatArray> {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        try {
            return decode(extractor)
        } finally {
            extractor.release()
        }
    }

    private fun decode(extractor: MediaExtractor): List<FloatArray> {
        val track = (0 until extractor.trackCount).firstOrNull {
            extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: throw IllegalArgumentException("找不到音訊軌")
        extractor.selectTrack(track)
        val format = extractor.getTrackFormat(track)
        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()

        var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channelCount = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        var floatPcm = false
        val samples = FloatList()
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        try {
            while (true) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val f = codec.outputFormat
                    sampleRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    channelCount = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    floatPcm = f.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                        f.getInteger(MediaFormat.KEY_PCM_ENCODING) == AudioFormat.ENCODING_PCM_FLOAT
                } else if (outIdx >= 0) {
                    val buf = codec.getOutputBuffer(outIdx)!!.order(ByteOrder.LITTLE_ENDIAN)
                    buf.position(info.offset)
                    buf.limit(info.offset + info.size)
                    if (floatPcm) {
                        val fb = buf.asFloatBuffer()
                        while (fb.hasRemaining()) samples.add(fb.get())
                    } else {
                        // 與 soundfile 相同：int16 / 32768
                        val sb = buf.asShortBuffer()
                        while (sb.hasRemaining()) samples.add(sb.get() / 32768f)
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) break
                }
            }
        } finally {
            codec.stop()
            codec.release()
        }

        val interleaved = samples.toArray()
        val frames = interleaved.size / channelCount
        return List(channelCount) { c ->
            val ch = FloatArray(frames) { interleaved[it * channelCount + c] }
            resample(ch, sampleRate)
        }
    }

    private tailrec fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

    private class FloatList {
        private var data = FloatArray(1 shl 16)
        private var size = 0
        fun add(v: Float) {
            if (size == data.size) data = data.copyOf(size * 2)
            data[size++] = v
        }
        fun toArray(): FloatArray = data.copyOf(size)
    }
}
