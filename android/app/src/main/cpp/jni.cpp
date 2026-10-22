#include <jni.h>

#include <string>

#include "engine.h"

static std::string to_string(JNIEnv *env, jstring s) {
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

// Java's modified UTF-8 mangles emoji, so text crosses the boundary as UTF-8 bytes.
static jbyteArray to_bytes(JNIEnv *env, const std::string &s) {
    jbyteArray arr = env->NewByteArray((jsize) s.size());
    env->SetByteArrayRegion(arr, 0, (jsize) s.size(), reinterpret_cast<const jbyte *>(s.data()));
    return arr;
}

static std::string from_bytes(JNIEnv *env, jbyteArray b) {
    jsize n = env->GetArrayLength(b);
    std::string out(n, '\0');
    env->GetByteArrayRegion(b, 0, n, reinterpret_cast<jbyte *>(out.data()));
    return out;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_handsfree_agent_LlamaEngine_nativeCreate(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(new Engine());
}

JNIEXPORT jstring JNICALL
Java_ai_handsfree_agent_LlamaEngine_nativeLoad(JNIEnv *env, jobject, jlong handle, jstring path, jint nCtx, jint nThreads) {
    std::string err = reinterpret_cast<Engine *>(handle)->load(to_string(env, path), nCtx, nThreads);
    return env->NewStringUTF(err.c_str());
}

JNIEXPORT jbyteArray JNICALL
Java_ai_handsfree_agent_LlamaEngine_nativeGenerate(JNIEnv *env, jobject, jlong handle, jbyteArray system,
                                                   jbyteArray user, jstring grammar, jint maxTokens) {
    std::string out = reinterpret_cast<Engine *>(handle)->generate(
            from_bytes(env, system), from_bytes(env, user), to_string(env, grammar), maxTokens);
    return to_bytes(env, out);
}

JNIEXPORT void JNICALL
Java_ai_handsfree_agent_LlamaEngine_nativeFree(JNIEnv *, jobject, jlong handle) {
    delete reinterpret_cast<Engine *>(handle);
}

}
