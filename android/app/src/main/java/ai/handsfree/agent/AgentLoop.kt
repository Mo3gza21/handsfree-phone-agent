package ai.handsfree.agent

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * See the screen, ask the model for one action, do it, repeat, until the model says done.
 * The model file is loaded once and shared by the voice service and the test screen.
 */
object AgentLoop {
    private const val TAG = "AgentLoop"
    private const val MAX_STEPS = 25
    private const val SETTLE_MS = 1200L

    private var engine: LlamaEngine? = null
    private val lock = Mutex()

    fun modelFile(context: Context) = File(context.filesDir, "model.gguf")

    private suspend fun engine(context: Context): LlamaEngine = withContext(Dispatchers.Default) {
        engine ?: LlamaEngine().also {
            it.load(modelFile(context).absolutePath)
            engine = it
        }
    }

    fun unload() {
        engine?.close()
        engine = null
    }

    /**
     * Carries out [goal]. [ask] speaks a question and returns the user's spoken answer;
     * [progress] reports each step for the screen/log. Returns what to say when finished.
     */
    suspend fun run(
        context: Context, goal: String,
        ask: suspend (String) -> String?,
        progress: (String) -> Unit = {},
    ): String = lock.withLock {
        val service = AgentAccessibilityService.instance
            ?: return "Turn on the Hands-Free Agent accessibility service first."
        if (!modelFile(context).exists()) return "Choose the model file in the app first."
        val llm = try {
            engine(context)
        } catch (e: Exception) {
            return "Could not load the model: ${e.message}"
        }

        var task = goal
        val history = mutableListOf<String>()
        var badReplies = 0
        var lastKey = ""
        var repeats = 0
        repeat(MAX_STEPS) {
            val screen = withContext(Dispatchers.Main) { service.readScreen() }
            if (screen == null) {
                delay(SETTLE_MS)
                return@repeat
            }
            val prompt = AgentPrompt.userMessage(task, history, screen.render())
            val reply = withContext(Dispatchers.Default) {
                llm.generate(AgentPrompt.SYSTEM_PROMPT, prompt, AgentPrompt.ACTION_GRAMMAR, 128)
            }
            Log.d(TAG, "screen:\n${screen.render()}\nreply: $reply")
            val action = AgentPrompt.parse(reply, screen.ids)
            if (action == null) {
                if (++badReplies >= 3) return "Sorry, I got confused on this screen."
                return@repeat
            }
            when (action.kind) {
                "done" -> return action.say.ifEmpty { "Done." }
                "ask_user" -> {
                    val answer = ask(action.question) ?: return "Okay, stopping."
                    // The model was trained on single requests, so fold the answer into the goal.
                    task = "$goal ($answer)"
                    return@repeat
                }
            }
            val label = screen.byId(action.id)?.label
            val key = "${screen.render().hashCode()}|$action"
            repeats = if (key == lastKey) repeats + 1 else 0
            lastKey = key
            if (repeats >= 2) return "I'm stuck repeating the same step, so I stopped."

            val step = AgentPrompt.describe(action, label)
            progress(step)
            val error = withContext(Dispatchers.Main) { service.perform(action, screen) }
            if (error != null) Log.w(TAG, "$step failed: $error")
            history.add(step)
            delay(if (action.kind == "open_app") SETTLE_MS * 2 else SETTLE_MS)
        }
        "That took too many steps, so I stopped."
    }
}
