package ai.handsfree.agent

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.graphics.Rect
import android.os.Bundle
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/** The agent's eyes and hands: reads the screen and performs actions on it. */
class AgentAccessibilityService : AccessibilityService() {

    private lateinit var reader: ScreenReader

    override fun onServiceConnected() {
        reader = ScreenReader(this)
        instance = this
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}

    fun readScreen(): ScreenSnapshot? = rootInActiveWindow?.let { reader.read(it) }

    /** Returns null on success, or a short reason it failed. Call on the main thread. */
    suspend fun perform(a: Action, screen: ScreenSnapshot): String? = when (a.kind) {
        "open_app" -> openApp(a.app)
        "tap" -> screen.byId(a.id)?.let { click(it.node, long = false) } ?: "no element ${a.id}"
        "long_press" -> screen.byId(a.id)?.let { click(it.node, long = true) } ?: "no element ${a.id}"
        "type" -> screen.byId(a.id)?.let { type(it.node, a.text) } ?: "no element ${a.id}"
        "enter" -> enter()
        "scroll" -> scroll(a.direction, screen)
        "back" -> if (performGlobalAction(GLOBAL_ACTION_BACK)) null else "back failed"
        "home" -> if (performGlobalAction(GLOBAL_ACTION_HOME)) null else "home failed"
        "wait" -> null
        else -> "not a phone action: ${a.kind}"
    }

    private fun openApp(name: String): String? {
        val pm = packageManager
        val apps = pm.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)
        val want = name.lowercase().trim()
        val labelled = apps.map { it to it.loadLabel(pm).toString().lowercase() }
        val match = labelled.firstOrNull { it.second == want }
            ?: labelled.firstOrNull { it.second.startsWith(want) || want.startsWith(it.second) }
            ?: labelled.firstOrNull { it.second.contains(want) || want.contains(it.second) }
            ?: return "no app called $name"
        val intent = pm.getLaunchIntentForPackage(match.first.activityInfo.packageName) ?: return "can't launch $name"
        startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED))
        return null
    }

    private suspend fun click(node: AccessibilityNodeInfo, long: Boolean): String? {
        val action = if (long) AccessibilityNodeInfo.ACTION_LONG_CLICK else AccessibilityNodeInfo.ACTION_CLICK
        var n: AccessibilityNodeInfo? = node
        while (n != null) {
            if ((if (long) n.isLongClickable else n.isClickable) && n.performAction(action)) return null
            n = n.parent
        }
        // Some apps draw their own buttons: fall back to a real touch on the element.
        val r = Rect().also { node.getBoundsInScreen(it) }
        return if (gesture(r.exactCenterX(), r.exactCenterY(), r.exactCenterX(), r.exactCenterY(), if (long) 800 else 60)) null
        else "tap failed"
    }

    private fun type(node: AccessibilityNodeInfo, text: String): String? {
        node.performAction(AccessibilityNodeInfo.ACTION_FOCUS)
        val args = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        return if (node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) null else "can't type there"
    }

    private fun enter(): String? {
        val focused = rootInActiveWindow?.findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: return "no text box focused"
        return if (focused.performAction(AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.id)) null
        else "enter failed"
    }

    private suspend fun scroll(direction: String, screen: ScreenSnapshot): String? {
        val target = screen.scrollables.maxByOrNull { Rect().also(it::getBoundsInScreen).let { r -> r.width() * r.height() } }
        val forward = direction == "down" || direction == "right"
        if (target != null && target.performAction(
                if (forward) AccessibilityNodeInfo.ACTION_SCROLL_FORWARD else AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
            )
        ) return null
        val m = resources.displayMetrics
        val (w, h) = m.widthPixels.toFloat() to m.heightPixels.toFloat()
        val ok = when (direction) {
            "down" -> gesture(w / 2, h * 0.75f, w / 2, h * 0.3f, 300)
            "up" -> gesture(w / 2, h * 0.3f, w / 2, h * 0.75f, 300)
            "right" -> gesture(w * 0.8f, h / 2, w * 0.2f, h / 2, 300)
            else -> gesture(w * 0.2f, h / 2, w * 0.8f, h / 2, 300)
        }
        return if (ok) null else "scroll failed"
    }

    private suspend fun gesture(x1: Float, y1: Float, x2: Float, y2: Float, ms: Long): Boolean =
        suspendCancellableCoroutine { cont ->
            val path = Path().apply { moveTo(x1, y1); lineTo(x2, y2) }
            val g = GestureDescription.Builder().addStroke(GestureDescription.StrokeDescription(path, 0, ms)).build()
            val sent = dispatchGesture(g, object : GestureResultCallback() {
                override fun onCompleted(d: GestureDescription?) { if (cont.isActive) cont.resume(true) }
                override fun onCancelled(d: GestureDescription?) { if (cont.isActive) cont.resume(false) }
            }, null)
            if (!sent && cont.isActive) cont.resume(false)
        }

    companion object {
        @Volatile
        var instance: AgentAccessibilityService? = null
    }
}
