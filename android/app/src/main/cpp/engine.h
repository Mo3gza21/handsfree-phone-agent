// On-device LLM engine: plain C++ over llama.cpp, wrapped for Kotlin in jni.cpp.
#pragma once

#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;

class Engine {
public:
    ~Engine();
    // Returns an empty string on success, otherwise the error.
    std::string load(const std::string &path, int n_ctx, int n_threads);
    // Greedy decode of the assistant reply, constrained by a GBNF grammar if one is given.
    std::string generate(const std::string &system, const std::string &user,
                         const std::string &grammar, int max_tokens);

private:
    std::string format_chat(const std::string &system, const std::string &user) const;
    bool decode_from(const std::vector<int> &tokens, size_t start);

    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    // Tokens currently in the KV cache. The system prompt is the same every turn,
    // so only the part after the shared prefix gets re-processed.
    std::vector<int> cached_;
};
