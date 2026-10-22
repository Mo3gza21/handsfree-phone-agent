package ai.handsfree.agent

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Setup screen: permissions, model file, language, and a typed test box. */
class MainActivity : Activity() {

    private val scope: CoroutineScope = MainScope()
    private lateinit var status: TextView
    private lateinit var log: TextView
    private lateinit var listenButton: Button
    private lateinit var langButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad, pad, pad) }
        fun button(text: String, onClick: () -> Unit) = Button(this).apply {
            this.text = text
            setOnClickListener { onClick() }
            col.addView(this)
        }

        status = TextView(this).apply { textSize = 16f }
        col.addView(status)
        button("1. Turn on accessibility (Hands-Free Agent)") {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }
        button("2. Allow microphone + notifications") {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.POST_NOTIFICATIONS), 1)
        }
        button("3. Choose the model file (.gguf)") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
                .setType("*/*"), PICK_MODEL)
        }
        langButton = button("") {
            Prefs.setLanguage(this, if (Prefs.language(this) == "ar-EG") "en-US" else "ar-EG")
            refresh()
        }
        val wake = EditText(this).apply {
            hint = "Wake word (empty = always listening)"
            setText(Prefs.wakeWord(this@MainActivity))
            col.addView(this)
        }
        listenButton = button("") {
            Prefs.setWakeWord(this, wake.text.toString())
            if (VoiceService.running) VoiceService.stop(this) else VoiceService.start(this)
            status.postDelayed({ refresh() }, 300)
        }

        col.addView(TextView(this).apply { text = "\nTry a command by typing:"; textSize = 15f })
        val command = EditText(this).apply { hint = "e.g. Open Settings and turn on Wi-Fi"; col.addView(this) }
        button("Run") {
            val goal = command.text.toString().trim()
            if (goal.isNotEmpty()) runTyped(goal)
        }
        log = TextView(this).apply { setTextIsSelectable(true) }
        col.addView(log)
        setContentView(ScrollView(this).apply { addView(col) })
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    private fun refresh() {
        val access = AgentAccessibilityService.instance != null
        val mic = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        val model = AgentLoop.modelFile(this)
        status.text = buildString {
            appendLine(if (access) "✅ Accessibility on" else "❌ Accessibility off")
            appendLine(if (mic) "✅ Microphone allowed" else "❌ Microphone not allowed")
            appendLine(if (model.exists()) "✅ Model: ${model.length() / 1_000_000} MB" else "❌ No model chosen")
        }
        langButton.text = "Language: " + if (Prefs.language(this) == "ar-EG") "Arabic (Egypt)" else "English"
        listenButton.text = if (VoiceService.running) "Stop hands-free" else "4. Start hands-free"
        listenButton.isEnabled = access && mic && model.exists()
    }

    private fun runTyped(goal: String) {
        log.text = "Goal: $goal\n"
        // Give the user a moment to see the run start, then let the agent work on whatever is on screen.
        moveTaskToBack(true)
        scope.launch {
            val result = AgentLoop.run(this@MainActivity, goal,
                ask = { q -> withContext(Dispatchers.Main) { log.append("Asked: $q\n") }; null },
                progress = { step -> runOnUiThread { log.append("• $step\n") } })
            log.append("Result: $result\n")
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        val uri: Uri = data?.data ?: return
        if (requestCode != PICK_MODEL || resultCode != RESULT_OK) return
        status.text = "Copying model… (about a minute)"
        listenButton.visibility = View.INVISIBLE
        scope.launch {
            val ok = withContext(Dispatchers.IO) {
                runCatching {
                    AgentLoop.unload()
                    val tmp = AgentLoop.modelFile(this@MainActivity).resolveSibling("model.part")
                    contentResolver.openInputStream(uri)!!.use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 20) } }
                    tmp.renameTo(AgentLoop.modelFile(this@MainActivity))
                }.getOrDefault(false)
            }
            listenButton.visibility = View.VISIBLE
            refresh()
            if (!ok) status.append("\nCopy failed — is there enough free space?")
        }
    }

    companion object {
        private const val PICK_MODEL = 7
    }
}
