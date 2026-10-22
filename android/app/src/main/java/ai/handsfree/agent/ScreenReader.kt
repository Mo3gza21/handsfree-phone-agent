package ai.handsfree.agent

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.view.accessibility.AccessibilityNodeInfo

/** One element as the model sees it: `[3] button "Send"`, plus the live node to act on. */
class UiElement(
    val id: Int, val role: String, val label: String, val checked: Boolean?,
    val focused: Boolean, val node: AccessibilityNodeInfo,
) {
    fun render(): String {
        val flags = buildList {
            if (checked != null) add(if (checked) "on" else "off")
            if (focused) add("focused")
        }
        val esc = label.replace("\"", "'")
        return "[$id] $role \"$esc\"" + if (flags.isEmpty()) "" else " (${flags.joinToString(", ")})"
    }
}

class ScreenSnapshot(val app: String, val elements: List<UiElement>, val scrollables: List<AccessibilityNodeInfo>) {
    val ids: Set<Int> = elements.map { it.id }.toSet()
    fun render(): String = (listOf("App: $app") + elements.map { it.render() }).joinToString("\n")
    fun byId(id: Int): UiElement? = elements.firstOrNull { it.id == id }
}

/**
 * Turns the accessibility tree into the compact screen text the model was trained on
 * (see pipeline/spec.py Screen.render). Ids are given in reading order from 1.
 */
class ScreenReader(private val context: Context) {

    private val homePackage: String? by lazy {
        val intent = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME)
        context.packageManager.resolveActivity(intent, PackageManager.MATCH_DEFAULT_ONLY)?.activityInfo?.packageName
    }

    fun read(root: AccessibilityNodeInfo): ScreenSnapshot {
        val out = mutableListOf<UiElement>()
        val scrollables = mutableListOf<AccessibilityNodeInfo>()
        visit(root, out, scrollables, inTabBar = false)
        return ScreenSnapshot(appName(root.packageName?.toString()), out, scrollables)
    }

    private fun appName(pkg: String?): String {
        if (pkg == null) return "Unknown"
        if (pkg == homePackage) return "Launcher"
        return try {
            val pm = context.packageManager
            pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString()
        } catch (e: PackageManager.NameNotFoundException) {
            pkg
        }
    }

    private fun visit(
        node: AccessibilityNodeInfo, out: MutableList<UiElement>,
        scrollables: MutableList<AccessibilityNodeInfo>, inTabBar: Boolean,
    ) {
        if (out.size >= MAX_ELEMENTS || !node.isVisibleToUser) return
        val cls = node.className?.toString() ?: ""
        if (node.isScrollable) scrollables.add(node)
        val tabBar = inTabBar || cls.contains("TabLayout") || cls.contains("TabWidget") || cls.contains("NavigationBar")
        val own = ownText(node)
        val actionable = node.isClickable || node.isLongClickable || node.isEditable || node.isCheckable

        if (actionable) {
            // A clickable row usually has no text itself: name it after what it contains,
            // e.g. `item "Ahmed, On my way, 10:32"`, and don't list those texts again.
            val label = own ?: childText(node).joinToString(", ").ifEmpty { idName(node) }
            if (label.isNotEmpty()) {
                val role = role(node, cls, tabBar, own == null)
                val checked = when {
                    node.isCheckable -> node.isChecked
                    role == "tab" -> node.isSelected
                    else -> null
                }
                out.add(UiElement(out.size + 1, role, label.take(MAX_LABEL), checked, node.isEditable && node.isFocused, node))
            }
            for (i in 0 until node.childCount) {
                val child = node.getChild(i) ?: continue
                if (hasActionable(child)) visit(child, out, scrollables, tabBar)
                else if (child.isScrollable) scrollables.add(child)
            }
            return
        }
        if (own != null && node.childCount == 0) {
            out.add(UiElement(out.size + 1, if (cls.contains("Image")) "image" else "text", own.take(MAX_LABEL), null, false, node))
        }
        for (i in 0 until node.childCount) node.getChild(i)?.let { visit(it, out, scrollables, tabBar) }
    }

    private fun role(node: AccessibilityNodeInfo, cls: String, tabBar: Boolean, composite: Boolean): String = when {
        node.isEditable || cls.contains("EditText") -> "edit"
        node.isCheckable && (cls.contains("Switch") || cls.contains("CheckBox") || cls.contains("Toggle")) -> "switch"
        tabBar || cls.contains("Tab") -> "tab"
        composite || cls.contains("ListView") || cls.contains("ViewGroup") || cls.contains("Layout") -> "item"
        else -> "button"
    }

    private fun ownText(node: AccessibilityNodeInfo): String? {
        val t = node.text?.toString()?.trim()
        if (!t.isNullOrEmpty()) return t
        val d = node.contentDescription?.toString()?.trim()
        if (!d.isNullOrEmpty()) return d
        if (node.isEditable) node.hintText?.toString()?.trim()?.takeIf { it.isNotEmpty() }?.let { return it }
        return null
    }

    private fun childText(node: AccessibilityNodeInfo, depth: Int = 0): List<String> {
        if (depth > 4) return emptyList()
        val parts = mutableListOf<String>()
        for (i in 0 until node.childCount) {
            val c = node.getChild(i) ?: continue
            if (!c.isVisibleToUser || c.isClickable) continue
            ownText(c)?.let { parts.add(it) } ?: parts.addAll(childText(c, depth + 1))
            if (parts.size >= 4) break
        }
        return parts
    }

    private fun hasActionable(node: AccessibilityNodeInfo, depth: Int = 0): Boolean {
        if (node.isClickable || node.isLongClickable || node.isEditable || node.isCheckable) return true
        if (depth > 6) return false
        for (i in 0 until node.childCount) {
            val c = node.getChild(i) ?: continue
            if (hasActionable(c, depth + 1)) return true
        }
        return false
    }

    private fun idName(node: AccessibilityNodeInfo): String =
        node.viewIdResourceName?.substringAfterLast('/')?.replace('_', ' ') ?: ""

    companion object {
        const val MAX_ELEMENTS = 80
        const val MAX_LABEL = 80
    }
}
