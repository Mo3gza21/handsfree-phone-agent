package ai.handsfree.agent

import org.json.JSONObject

/**
 * Mirror of pipeline/spec.py. The model was trained on exactly these strings, so any
 * change here must be made there too (tests/test_spec_parity.py compares them).
 */
object AgentPrompt {

    val SYSTEM_PROMPT = """You control an Android phone for a user who cannot use their hands.
Each turn you get the user's goal, the actions you already took, and the current screen.
Reply with exactly one JSON object: the next action. No other text.
Actions:
{"action":"open_app","app":"<app name>"}
{"action":"tap","id":<element id>}
{"action":"long_press","id":<element id>}
{"action":"type","id":<element id>,"text":"<text>"}
{"action":"enter"}
{"action":"scroll","direction":"up|down|left|right"}
{"action":"back"}
{"action":"home"}
{"action":"wait"}
{"action":"ask_user","question":"<question>"}
{"action":"done","say":"<short reply to the user>"}
Only use ids that appear on the screen. If the goal is ambiguous, ask_user. If the goal is complete, done."""

    val ACTION_GRAMMAR = """root ::= "{" ws "\"action\"" ws ":" ws body ws "}"
body ::= (
         "\"open_app\"" ws "," ws "\"app\"" ws ":" ws str
       | "\"tap\"" ws "," ws "\"id\"" ws ":" ws int
       | "\"long_press\"" ws "," ws "\"id\"" ws ":" ws int
       | "\"type\"" ws "," ws "\"id\"" ws ":" ws int ws "," ws "\"text\"" ws ":" ws str
       | "\"enter\"" | "\"back\"" | "\"home\"" | "\"wait\""
       | "\"scroll\"" ws "," ws "\"direction\"" ws ":" ws ("\"up\"" | "\"down\"" | "\"left\"" | "\"right\"")
       | "\"ask_user\"" ws "," ws "\"question\"" ws ":" ws str
       | "\"done\"" ws "," ws "\"say\"" ws ":" ws str
       )
str ::= "\"" ([^"\\\n] | "\\" ["\\/bfnrtu])* "\""
int ::= [0-9]+
ws ::= " "?
"""

    private val FIELDS = mapOf(
        "open_app" to listOf("app"), "tap" to listOf("id"), "long_press" to listOf("id"),
        "type" to listOf("id", "text"), "enter" to emptyList(), "scroll" to listOf("direction"),
        "back" to emptyList(), "home" to emptyList(), "wait" to emptyList(),
        "ask_user" to listOf("question"), "done" to listOf("say"),
    )

    fun userMessage(goal: String, history: List<String>, screen: String): String {
        val hist = if (history.isEmpty()) "none"
        else history.mapIndexed { i, h -> "${i + 1}. $h" }.joinToString("\n")
        return "Goal: $goal\nPrevious actions:\n$hist\nScreen:\n$screen"
    }

    /** Same wording as spec.describe_action / records.history_entry. */
    fun describe(a: Action, label: String?): String = when (a.kind) {
        "open_app" -> "open_app ${a.app}"
        "tap", "long_press" -> "${a.kind} ${a.id}" + (if (!label.isNullOrEmpty()) " \"$label\"" else "")
        "type" -> "type ${a.id} \"${a.text}\""
        "scroll" -> "scroll ${a.direction}"
        "ask_user" -> "ask_user \"${a.question}\""
        else -> a.kind
    }

    fun parse(reply: String, ids: Set<Int>): Action? {
        val start = reply.indexOf('{')
        if (start < 0) return null
        val obj = try {
            JSONObject(reply.substring(start, reply.lastIndexOf('}') + 1))
        } catch (e: Exception) {
            return null
        }
        val kind = obj.optString("action")
        val fields = FIELDS[kind] ?: return null
        if (fields.any { !obj.has(it) }) return null
        val id = if ("id" in fields) obj.optInt("id", -1).takeIf { it in ids } ?: return null else -1
        val dir = obj.optString("direction")
        if (kind == "scroll" && dir !in setOf("up", "down", "left", "right")) return null
        return Action(kind, id, obj.optString("app"), obj.optString("text"), dir,
            obj.optString("question"), obj.optString("say"))
    }
}

data class Action(
    val kind: String, val id: Int = -1, val app: String = "", val text: String = "",
    val direction: String = "", val question: String = "", val say: String = "",
)
