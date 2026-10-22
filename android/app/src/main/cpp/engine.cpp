#include "engine.h"

#include <algorithm>

#include "llama.h"

Engine::~Engine() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
}

std::string Engine::load(const std::string &path, int n_ctx, int n_threads) {
    static bool backend_ready = false;
    if (!backend_ready) {
        llama_backend_init();
        backend_ready = true;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (!model_) return "could not load model file: " + path;
    vocab_ = llama_model_get_vocab(model_);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) return "could not create context";
    return "";
}

std::string Engine::format_chat(const std::string &system, const std::string &user) const {
    llama_chat_message msgs[2] = {{"system", system.c_str()}, {"user", user.c_str()}};
    const char *tmpl = llama_model_chat_template(model_, nullptr);
    if (tmpl) {
        std::vector<char> buf(2 * (system.size() + user.size()) + 512);
        int n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int) buf.size());
        if (n > (int) buf.size()) {
            buf.resize(n);
            n = llama_chat_apply_template(tmpl, msgs, 2, true, buf.data(), (int) buf.size());
        }
        if (n > 0) return std::string(buf.data(), n);
    }
    // Fallback: ChatML, which the default Qwen base model uses.
    return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user +
           "<|im_end|>\n<|im_start|>assistant\n";
}

bool Engine::decode_from(const std::vector<int> &tokens, size_t start) {
    const size_t n_batch = llama_n_batch(ctx_);
    for (size_t i = start; i < tokens.size(); i += n_batch) {
        int n = (int) std::min(n_batch, tokens.size() - i);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data() + i), n);
        if (llama_decode(ctx_, batch) != 0) return false;
    }
    return true;
}

std::string Engine::generate(const std::string &system, const std::string &user,
                             const std::string &grammar, int max_tokens) {
    if (!ctx_) return "";
    const std::string prompt = format_chat(system, user);
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n = llama_tokenize(vocab_, prompt.c_str(), (int) prompt.size(), tokens.data(), (int) tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab_, prompt.c_str(), (int) prompt.size(), tokens.data(), (int) tokens.size(), true, true);
    }
    tokens.resize(std::max(n, 0));
    if (tokens.empty() || tokens.size() + max_tokens > llama_n_ctx(ctx_)) return "";

    // Keep the shared prefix (system prompt, usually) and drop everything after it.
    size_t keep = 0;
    while (keep < cached_.size() && keep < tokens.size() - 1 && cached_[keep] == tokens[keep]) keep++;
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1)) {
        llama_memory_clear(mem, true);
        keep = 0;
    }
    cached_.assign(tokens.begin(), tokens.begin() + keep);
    if (!decode_from(tokens, keep)) {
        llama_memory_clear(mem, true);
        cached_.clear();
        return "";
    }
    cached_ = tokens;

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!grammar.empty()) {
        llama_sampler *g = llama_sampler_init_grammar(vocab_, grammar.c_str(), "root");
        if (g) llama_sampler_chain_add(smpl, g);
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string out;
    char piece[256];
    for (int i = 0; i < max_tokens; i++) {
        llama_token tok = llama_sampler_sample(smpl, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, tok)) break;
        int len = llama_token_to_piece(vocab_, tok, piece, sizeof(piece), 0, false);
        if (len > 0) out.append(piece, len);
        cached_.push_back(tok);
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx_, batch) != 0) break;
    }
    llama_sampler_free(smpl);
    return out;
}
