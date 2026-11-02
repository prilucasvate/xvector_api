package tw.xvector.demo

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.OpenableColumns
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import tw.xvector.Audio
import tw.xvector.SpeakerEncoder
import tw.xvector.demo.databinding.ActivityMainBinding
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private lateinit var b: ActivityMainBinding
    private var encoder: SpeakerEncoder? = null
    private val worker = Executors.newSingleThreadExecutor()
    private val embeddings = arrayOfNulls<FloatArray>(2)
    private var recorder: Recorder? = null
    private var pendingSlot = 0

    private val slots by lazy {
        listOf(
            Slot(b.nameA, b.metaA, b.pickA, b.recA),
            Slot(b.nameB, b.metaB, b.pickB, b.recB),
        )
    }

    private val pickAudio = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) analyzeUri(pendingSlot, uri)
    }

    private val askMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startRecording(pendingSlot) else status("需要麥克風權限才能錄音")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)

        slots.forEachIndexed { i, slot ->
            slot.pick.setOnClickListener {
                pendingSlot = i
                pickAudio.launch("audio/*")
            }
            slot.rec.setOnClickListener { toggleRecording(i) }
        }
        b.snorm.setOnCheckedChangeListener { _, _ -> render() }
        setButtonsEnabled(false)

        worker.execute {
            try {
                val e = SpeakerEncoder.fromAssets(this, "xvector_v14.onnx", "cohort.bin")
                runOnUiThread {
                    encoder = e
                    setButtonsEnabled(true)
                    status("模型已載入，請選擇或錄製兩段音訊。")
                }
            } catch (e: Exception) {
                runOnUiThread { status("模型載入失敗：${e.message}") }
            }
        }
    }

    override fun onDestroy() {
        recorder?.stop()
        worker.shutdown()
        encoder?.close()
        super.onDestroy()
    }

    private fun analyzeUri(i: Int, uri: Uri) {
        val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { if (it.moveToFirst()) it.getString(0) else null } ?: uri.lastPathSegment ?: "音檔"
        analyze(i, name) { Audio.preprocess(Audio.decode(this, uri)) }
    }

    /** 解碼與推論在背景執行緒進行，完成後更新畫面 */
    private fun analyze(i: Int, name: String, load: () -> FloatArray) {
        val enc = encoder ?: return
        slots[i].name.text = name
        slots[i].meta.text = "分析中…"
        embeddings[i] = null
        b.result.visibility = android.view.View.GONE
        worker.execute {
            try {
                val wav = load()
                val t0 = SystemClock.elapsedRealtime()
                val e = enc.embed(wav)
                val ms = SystemClock.elapsedRealtime() - t0
                runOnUiThread {
                    embeddings[i] = e
                    slots[i].meta.text = "%.1f 秒・推論 %d ms".format(wav.size / Audio.SAMPLE_RATE.toFloat(), ms)
                    status("")
                    render()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    slots[i].meta.text = ""
                    status("無法處理 $name：${e.message}")
                }
            }
        }
    }

    private fun render() {
        val e1 = embeddings[0] ?: return
        val e2 = embeddings[1] ?: return
        val s = encoder!!.score(e1, e2, useSnorm = b.snorm.isChecked)
        b.result.visibility = android.view.View.VISIBLE
        b.verdict.text = if (s.isSamePerson) "判定：同一人" else "判定：不同人"
        b.verdict.setTextColor(ContextCompat.getColor(this, if (s.isSamePerson) R.color.ok else R.color.bad))
        b.raw.text = "%.4f".format(s.rawScore)
        b.sn.text = s.snormScore?.let { "%.2f".format(it) } ?: "–"
        b.th.text = s.threshold.toString()
    }

    private fun toggleRecording(i: Int) {
        val r = recorder
        if (r != null) {
            if (r.slot == i) r.stop()
            return
        }
        pendingSlot = i
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startRecording(i)
        } else {
            askMic.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startRecording(i: Int) {
        val r = Recorder(i)
        recorder = r
        slots.forEachIndexed { j, s ->
            s.pick.isEnabled = false
            s.rec.isEnabled = j == i
        }
        status("錄音中，建議至少說 3 秒…")
        r.start { seconds -> runOnUiThread { slots[i].rec.text = "■ 停止 ${seconds}s" } }
    }

    private fun onRecorded(i: Int, samples: FloatArray) {
        recorder = null
        setButtonsEnabled(true)
        slots[i].rec.text = "● 重新錄音"
        if (samples.size < Audio.SAMPLE_RATE / 2) {
            status("錄音太短，請再試一次")
            return
        }
        analyze(i, "錄音 ${"AB"[i]}") { Audio.preprocess(listOf(samples)) }
    }

    private fun setButtonsEnabled(enabled: Boolean) = slots.forEach {
        it.pick.isEnabled = enabled
        it.rec.isEnabled = enabled
    }

    private fun status(text: String) {
        b.status.text = text
    }

    private class Slot(val name: TextView, val meta: TextView, val pick: Button, val rec: Button)

    /** 以 AudioRecord 直接錄製 16kHz 單聲道 float，不需再解碼或重新取樣 */
    private inner class Recorder(val slot: Int) {
        @Volatile private var running = true

        @SuppressLint("MissingPermission")
        fun start(onTick: (Int) -> Unit) = thread(name = "recorder") {
            val minBuf = AudioRecord.getMinBufferSize(
                Audio.SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT,
            )
            // VOICE_RECOGNITION 會關閉大部分的降噪與自動增益，較接近訓練資料
            val record = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION, Audio.SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT, maxOf(minBuf, 4096) * 4,
            )
            val out = ArrayList<FloatArray>()
            val buf = FloatArray(1600)
            var total = 0
            var lastTick = -1
            try {
                record.startRecording()
                while (running) {
                    val n = record.read(buf, 0, buf.size, AudioRecord.READ_BLOCKING)
                    if (n <= 0) break
                    out.add(buf.copyOf(n))
                    total += n
                    val sec = total / Audio.SAMPLE_RATE
                    if (sec != lastTick) {
                        lastTick = sec
                        onTick(sec)
                    }
                }
            } finally {
                record.stop()
                record.release()
            }
            val samples = FloatArray(total)
            var off = 0
            for (chunk in out) {
                System.arraycopy(chunk, 0, samples, off, chunk.size)
                off += chunk.size
            }
            runOnUiThread { onRecorded(slot, samples) }
        }

        fun stop() {
            running = false
        }
    }
}
