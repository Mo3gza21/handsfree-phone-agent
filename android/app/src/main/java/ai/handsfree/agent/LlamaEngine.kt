package ai.handsfree.agent

/** Thin Kotlin face of cpp/engine.cpp. One instance per loaded model; not thread-safe. */
class LlamaEngine {
    private var handle = nativeCreate()

    /** Throws if the model file can't be loaded. */
    fun load(path: String, nCtx: Int = 2048, threads: Int = defaultThreads()) {
        val err = nativeLoad(handle, path, nCtx, threads)
        if (err.isNotEmpty()) throw IllegalStateException(err)
    }

    fun generate(system: String, user: String, grammar: String = "", maxTokens: Int = 128): String =
        String(nativeGenerate(handle, system.toByteArray(), user.toByteArray(), grammar, maxTokens), Charsets.UTF_8)

    fun close() {
        if (handle != 0L) nativeFree(handle)
        handle = 0L
    }

    private external fun nativeCreate(): Long
    private external fun nativeLoad(handle: Long, path: String, nCtx: Int, threads: Int): String
    private external fun nativeGenerate(handle: Long, system: ByteArray, user: ByteArray, grammar: String, maxTokens: Int): ByteArray
    private external fun nativeFree(handle: Long)

    companion object {
        init {
            System.loadLibrary("phoneagent")
        }

        // Phones mix big and little cores; using only the big ones is usually fastest.
        fun defaultThreads() = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)
    }
}
