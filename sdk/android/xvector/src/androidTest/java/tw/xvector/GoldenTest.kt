package tw.xvector

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** 與 tools/make_golden.py 以 PyTorch 產生的結果比對。 */
@RunWith(AndroidJUnit4::class)
class GoldenTest {
    private val context = InstrumentationRegistry.getInstrumentation().context
    private val golden = JSONObject(context.assets.open("golden.json").bufferedReader().readText())

    private fun asset(name: String): String {
        val file = File(context.cacheDir, name)
        context.assets.open(name).use { input -> file.outputStream().use { input.copyTo(it) } }
        return file.path
    }

    private fun goldenEmbedding(name: String): FloatArray {
        val arr = golden.getJSONArray(name)
        return FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
    }

    @Test
    fun matchesPyTorch() {
        SpeakerEncoder.fromAssets(context, "xvector_v14.onnx", "cohort.bin").use { encoder ->
            val emb = listOf("c", "d").associateWith { name ->
                val e = encoder.embed(Audio.preprocess(Audio.decode(asset("$name.wav"))))
                val cos = SpeakerEncoder.dot(e, goldenEmbedding(name))
                println("[GoldenTest] $name cosine vs PyTorch = $cos")
                assertTrue("$name cosine $cos", cos > 0.9999f)
                e
            }
            val s = encoder.score(emb.getValue("c"), emb.getValue("d"))
            val ref = golden.getJSONObject("score")
            println("[GoldenTest] Kotlin  = $s")
            println("[GoldenTest] PyTorch = $ref")
            assertEquals(ref.getDouble("raw_score").toFloat(), s.rawScore, 1e-3f)
            assertEquals(ref.getDouble("snorm_score").toFloat(), s.snormScore!!, 1e-2f)
        }
    }
}
