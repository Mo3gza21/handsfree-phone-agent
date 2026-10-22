package ai.handsfree.agent

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Bundle
import android.os.IBinder
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.Locale
import kotlin.coroutines.resume

/**
 * Always-on listener. Say the wake word followed by a command ("Maya, call Mom"),
 * or just the command when the wake word is turned off. Runs as a foreground
 * service so Android lets it keep the microphone.
 */
class VoiceService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var recognizer: SpeechRecognizer
    private lateinit var tts: TextToSpeech
    private var busy = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        startForeground(1, notification("Listening"), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
        recognizer = SpeechRecognizer.createSpeechRecognizer(this)
        tts = TextToSpeech(this) { tts.language = Locale.forLanguageTag(Prefs.language(this)) }
        running = true
        scope.launch { listenLoop() }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) stopSelf()
        return START_STICKY
    }

    override fun onDestroy() {
        running = false
        scope.cancel()
        recognizer.destroy()
        tts.shutdown()
        super.onDestroy()
    }

    private suspend fun listenLoop() {
        while (true) {
            val heard = listen() ?: continue
            val command = Prefs.stripWakeWord(this, heard) ?: continue
            busy = true
            updateNotification("Working: $command")
            val reply = AgentLoop.run(this, command, ask = { q -> speak(q); listen(timeoutRetries = 2) })
            speak(reply)
            busy = false
            updateNotification("Listening")
        }
    }

    /** One recognition attempt; null on silence or error. */
    private suspend fun listen(timeoutRetries: Int = 0): String? {
        repeat(timeoutRetries + 1) {
            val text = recognizeOnce()
            if (!text.isNullOrBlank()) return text
            delay(300)
        }
        return null
    }

    private suspend fun recognizeOnce(): String? = suspendCancellableCoroutine { cont ->
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Prefs.language(this@VoiceService))
            putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        }
        recognizer.setRecognitionListener(object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                if (cont.isActive) cont.resume(text)
            }
            override fun onError(error: Int) { if (cont.isActive) cont.resume(null) }
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        recognizer.startListening(intent)
        cont.invokeOnCancellation { recognizer.cancel() }
    }

    /** Speaks and waits until finished, so the mic doesn't hear the phone talking. */
    private suspend fun speak(text: String) = suspendCancellableCoroutine { cont ->
        val id = System.nanoTime().toString()
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) { if (utteranceId == id && cont.isActive) cont.resume(Unit) }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) { if (cont.isActive) cont.resume(Unit) }
        })
        if (tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, id) != TextToSpeech.SUCCESS && cont.isActive) cont.resume(Unit)
    }

    private fun notification(text: String): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Voice control", NotificationManager.IMPORTANCE_LOW))
        val stop = PendingIntent.getService(this, 0, Intent(this, VoiceService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Hands-Free Agent")
            .setContentText(text)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }

    private fun updateNotification(text: String) =
        getSystemService(NotificationManager::class.java).notify(1, notification(text))

    companion object {
        private const val CHANNEL = "voice"
        private const val ACTION_STOP = "stop"

        @Volatile
        var running = false
            private set

        fun start(context: Context) = context.startForegroundService(Intent(context, VoiceService::class.java))
        fun stop(context: Context) = context.stopService(Intent(context, VoiceService::class.java))
    }
}

object Prefs {
    private fun prefs(c: Context) = c.getSharedPreferences("agent", Context.MODE_PRIVATE)

    fun language(c: Context): String = prefs(c).getString("lang", "ar-EG")!!
    fun setLanguage(c: Context, tag: String) = prefs(c).edit().putString("lang", tag).apply()
    fun wakeWord(c: Context): String = prefs(c).getString("wake", "Maya")!!
    fun setWakeWord(c: Context, w: String) = prefs(c).edit().putString("wake", w).apply()

    /** The command after the wake word, or null if the wake word wasn't said. Empty wake word = always on. */
    fun stripWakeWord(c: Context, heard: String): String? {
        val wake = wakeWord(c).trim()
        if (wake.isEmpty()) return heard.trim()
        val forms = (if (wake.equals("Maya", ignoreCase = true)) listOf(wake, "مايا") else listOf(wake)).map { it.lowercase() }
        val lower = heard.lowercase()
        for (w in forms) {
            val i = lower.indexOf(w)
            if (i >= 0) return heard.substring(i + w.length).trim(' ', ',', '،', '.').ifEmpty { null }
        }
        return null
    }
}
