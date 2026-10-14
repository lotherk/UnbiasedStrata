// frontend/main.cpp - UnbiasedStrata's front end: tokenizer, chat template and an OpenAI-style HTTP endpoint in
// front of the unmodified Strata engine.
//
// One binary, two roles.  Started normally it is the front end: it starts a copy of itself with `--engine`, which
// runs the engine's own program (frontend/engine_entry.cpp) in serve mode, and talks to it over a pipe in the
// engine's line protocol (token ids in, token ids out).  That is the process model upstream's Python server uses,
// so the engine runs exactly as upstream tested it.
//
// The tokenizer and the chat template come from llama.cpp (the pinned submodule): the GGUF's own vocabulary and
// its own Jinja template.  Nothing here opens an outgoing connection.
#include "engine.hpp"
#include "vision.hpp"
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int strata_engine_main(int argc, char** argv);

namespace {

// the model id reported on /v1/models and in every completion: the GGUF's base name (`Qwen3.6-35B-A3B-UD-IQ3_XXS`,
// `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` for shard 1 without its `-00001-of-00002`), set once the options are parsed
std::string g_model_name = "unbiased-strata";
std::string model_name_of(const std::string& path) {
    std::string b = path.substr(path.find_last_of('/') == std::string::npos ? 0 : path.find_last_of('/') + 1);
    if (b.size() > 5 && b.compare(b.size() - 5, 5, ".gguf") == 0) b.resize(b.size() - 5);
    const size_t at = b.rfind("-00001-of-");
    if (at != std::string::npos) b.resize(at);
    return b.empty() ? std::string("unbiased-strata") : b;
}

// The engine's serve loop refuses a request unless prompt + max_new + this margin fits --ctx
// (src/program/generate.cpp:6403): the last cells are the engine's own, not ours to hand out.
constexpr long long kEngineReserve = 8;

// #123: what is injected before the thinking's end tag when reasoning_budget_tokens is reached.  The wording is
// switchable (--reasoning-budget-message, or the request's reasoning_budget_message); the default is upstream
// Strata's own wrap-up line.
const char* const kDefaultReasoningBudgetMessage = "I have thought about this long enough; time to give my answer.";

struct Options {
    std::string model;                  // shard 1 of the GGUF
    std::string pack;                   // the prepared pack directory
    std::string mtp;                    // the prepared draft-head directory
    std::string mmproj;
    int vision_tokens = 1024;
    std::string host = "127.0.0.1";
    int port = 8080;
    long long ctx = 32768;
    int threads = 0;                    // CPU expert workers; 0 = the engine's default (every physical core)
    bool fit_max_tokens = true;         // shorten a request's max_tokens to the room left instead of refusing it
    long long vram_cap_mib = 0;         // --vram-cap: the engine limits itself to a smaller card's VRAM
    double ram_cap_gib = 0.0;           // --ram-cap: bounds the engine's RAM expert tier and pinned arena
    long long reasoning_budget = 0;     // cap the thinking of every request (0: no cap; a request can set its own)
    std::string reasoning_budget_message = kDefaultReasoningBudgetMessage;   // injected before the end tag
    // Sampling defaults: a request's own fields win over these.  Temperature <= 0 keeps the documented default
    // (greedy) and emits no filters, exactly as before; the rest are the engine's own neutral values, so an
    // unconfigured front end changes nothing.
    double temperature = 0.0;           // > 0 samples; 0 (default) is greedy
    double top_p = 0.95;
    long long top_k = 20;               // the engine's sampler takes 1..64
    double min_p = 0.0;                 // 0 disables
    long long repeat_last_n = 0;        // penalty window; 0 with a penalty set falls back to 64
    double repeat_penalty = 1.0;        // 1.0 disables
    double presence_penalty = 0.0;      // 0.0 disables
    double frequency_penalty = 0.0;     // 0.0 disables
    // DRY (llama.cpp's `dry` sampler): breaks looping generations by penalising a token that would extend a
    // repeat longer than `dry_allowed_length`.  Off unless the multiplier and the window are both set.
    double dry_multiplier = 0.0;        // 0.0 disables
    double dry_base = 1.75;             // the exponent base (>= 1)
    long long dry_allowed_length = 2;   // repeats of this length or less are free
    long long dry_penalty_last_n = 0;   // the DRY window in tokens (0 disables)
    std::vector<std::string> dry_breakers;   // restart sequences; empty uses "\n", ":", "\"", "*"
    long long seed = -1;                // -1: a fresh random seed per request
    std::vector<std::string> extra;     // after `--`: passed to the engine as they are
};

void usage() {
    std::fprintf(stderr,
        "usage: unbiased-strata --model SHARD1.gguf --pack DIR --mtp DIR [options] [-- engine options]\n\n"
        "  --model PATH    the model's first GGUF shard (...-00001-of-00002.gguf); the second must be beside it\n"
        "  --pack DIR      the prepared pack for that model\n"
        "  --mtp DIR       the prepared draft head\n"
        "  --mmproj PATH   enable image input using this vision encoder (CPU)\n"
        "  --vision-tokens N  maximum tokens per image (default 1024)\n"
        "  --ctx N         context length in tokens (default 32768)\n"
        "  --fit-max-tokens / --no-fit-max-tokens\n"
        "                  shorten a request's max_tokens to the room left instead of refusing it (default: on)\n"
        "  --reasoning-budget-tokens N\n"
        "                  cap the thinking of every request at N tokens (default: 0, no cap; a request's own\n"
        "                  reasoning_budget_tokens wins)\n"
        "  --reasoning-budget-message TEXT\n"
        "                  injected before the end-of-thinking tag when the budget is reached (default: a short\n"
        "                  wrap-up line; a request's own reasoning_budget_message wins)\n"
        "  --threads N     CPU threads for the experts the GPU does not hold (default: every physical core,\n"
        "                  or half the CPU quota inside a container that has one)\n"
        "  --host ADDR     address to listen on (default 127.0.0.1)\n"
        "  --port N        port to listen on (default 8080)\n"
        "  --vram-cap MIB  emulate a smaller card: the engine uses at most MIB of VRAM (a 1080 Ti as 6 GB: 5722)\n"
        "  --ram-cap GIB   bound the engine's RAM expert tier and pinned arena to GIB\n\n"
        "Sampling defaults (a request's own temperature/top_p/top_k/min_p/seed/presence_penalty/\n"
        "frequency_penalty/repeat_penalty/repeat_last_n wins; greedy unless the effective temperature is > 0):\n"
        "  --temperature F sampling temperature (default 0 = greedy)\n"
        "  --top-k N       top-k shortlist, 1..64 (default 20)\n"
        "  --top-p F       nucleus probability, 0..1 (default 0.95)\n"
        "  --min-p F       min-p floor, 0..1 (default 0, off)\n"
        "  --repeat-last-n N   repetition-penalty window in tokens (default 0; a penalty set here or in a\n"
        "                  request forces at least 64)\n"
        "  --repeat-penalty F  repetition penalty, > 0 (default 1.0, off)\n"
        "  --presence-penalty F  presence penalty (default 0.0, off)\n"
        "  --frequency-penalty F frequency penalty (default 0.0, off)\n"
        "  --dry-multiplier F  DRY repetition penalty; 0 disables (default 0.0)\n"
        "  --dry-base F    DRY exponent base, >= 1 (default 1.75)\n"
        "  --dry-allowed-length N  free repeat length before DRY applies (default 2)\n"
        "  --dry-penalty-last-n N  the DRY window in tokens; 0 disables (default 0)\n"
        "  --dry-breakers LIST  restart sequences as literal text separated by '|' (default \n:|\"|*)\n"
        "  --seed N        sampling seed (default: a fresh random seed per request)\n\n"
        "Serves POST /v1/chat/completions (OpenAI style, with streaming), GET /v1/models and GET /health (with the\n"
        "engine's expert tiers and per-token counters: VRAM slots, RAM copies, mapped experts, hits, warm hints,\n"
        "major page faults).\n");
}

bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

// ---------------------------------------------------------------------------------------------------- the engine

using strata_frontend::Engine;
using strata_frontend::GenResult;

// ------------------------------------------------------------------------------------------------ text handling

/// The length of the longest prefix of `s` that does not end inside a UTF-8 character.
size_t utf8_complete_prefix(const std::string& s) {
    size_t i = s.size();
    int back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = (unsigned char) s[i - 1];
        ++back;
        if ((c & 0xC0) == 0x80) { --i; continue; }             // a continuation byte: keep looking for its lead
        const int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        return need > back ? i - 1 : s.size();
    }
    return s.size();
}

std::string make_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    char b[40];
    std::snprintf(b, sizeof b, "chatcmpl-%016llx", (unsigned long long) rng());
    return b;
}

long long now_s() {
    return (long long) std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void json_error(httplib::Response& res, int status, const std::string& message, const char* type) {
    common_json e = common_json::object();
    e["error"] = common_json::object();
    e["error"]["message"] = message;
    e["error"]["type"] = type;
    res.status = status;
    res.set_content(e.dump_safe(), "application/json");
}

// ------------------------------------------------------------------------------------------------- the request

struct Prepared {
    std::shared_ptr<strata_frontend::ImageFile> images;
    std::vector<llama_token> ids;
    std::string header;        // "GEN <max_new> key=value ... " - the ids follow
    std::string gen_prefix;    // "GEN " / "GENI ": the header without max_new and the keys
    std::string gen_suffix;    // the sampling keys (and the image path): the rest of the header
    std::string request_id;
    common_chat_parser_params parser;   // how to split the output into reasoning, answer and tool calls
    std::vector<std::string> thinking_end_tags;   // from the chat template: what ends the reasoning block
    std::string reasoning_budget_message;         // injected before the end tag at the budget
    bool thinking = true;                         // the template was rendered with thinking on
    long long reasoning_budget = 0;               // #123: cap the thinking at this many tokens (0: off)
    bool stream = false;
    long long max_new = 0;
};

/// The sampling keys of one request's engine header: the launch-time defaults (`opt`) with the request's own
/// OpenAI / llama.cpp fields winning.  Greedy is preserved: an effective temperature <= 0 emits no temperature
/// and no filters, and the penalties and their window are emitted only when a penalty is actually set (a
/// non-neutral value).  The order is fixed, so the header is reproducible for a given input.
std::string sampling_keys(const common_json& body, const Options& opt, const llama_vocab* vocab) {
    auto number = [&](const char* key, double fallback) {
        return body.contains(key) && body.at(key).is_number() ? body.at(key).get<double>() : fallback;
    };
    auto integer = [&](const char* key, long long fallback) {
        return body.contains(key) && body.at(key).is_number() ? body.at(key).get<long long>() : fallback;
    };
    auto text = [](double v) { char b[32]; std::snprintf(b, sizeof b, "%g", v); return std::string(b); };

    std::string keys;
    const double temperature = number("temperature", opt.temperature);
    if (temperature > 0.0) {
        const double top_p = number("top_p", opt.top_p);
        const long long top_k = std::max(1LL, std::min(64LL, integer("top_k", opt.top_k)));
        const double min_p = number("min_p", opt.min_p);
        const unsigned long long seed = body.contains("seed") && body.at("seed").is_number()
                                            ? (unsigned long long) body.at("seed").get<long long>()
                                            : (opt.seed >= 0 ? (unsigned long long) opt.seed : (unsigned long long) std::random_device{}());
        keys += " temperature=" + text(temperature) + " top_p=" + text(top_p) +
                " top_k=" + std::to_string(top_k) + " seed=" + std::to_string(seed);
        if (min_p > 0.0) keys += " min_p=" + text(min_p);
    }
    // llama.cpp's names: presence_penalty, frequency_penalty, repeat_penalty; repeat_last_n is the window.
    const double penalty_repeat = number("repeat_penalty", opt.repeat_penalty);
    const double penalty_freq = number("frequency_penalty", opt.frequency_penalty);
    const double penalty_present = number("presence_penalty", opt.presence_penalty);
    if (penalty_repeat != 1.0 || penalty_freq != 0.0 || penalty_present != 0.0) {
        if (penalty_repeat != 1.0) keys += " penalty_repeat=" + text(penalty_repeat);
        if (penalty_freq != 0.0) keys += " penalty_freq=" + text(penalty_freq);
        if (penalty_present != 0.0) keys += " penalty_present=" + text(penalty_present);
        // a penalty with a zero window is a no-op in the sampler, so give it llama.cpp's default of 64
        const long long window = integer("repeat_last_n", integer("penalty_last_n", opt.repeat_last_n));
        keys += " penalty_last_n=" + std::to_string(window > 0 ? window : 64);
    }
    // DRY (llama.cpp's `dry`): emitted only when it is actually on (multiplier and window both set).  The restart
    // sequences are tokenized here - the engine has no tokenizer - and sent as ids ("id,id|id|...", sequences by
    // '|'); a request's own `dry_sequence_breakers` array wins over the launch-time list, else llama.cpp's default.
    const double dry_multiplier = number("dry_multiplier", opt.dry_multiplier);
    const long long dry_window = integer("dry_penalty_last_n", opt.dry_penalty_last_n);
    if (dry_multiplier != 0.0 && dry_window != 0) {
        const double dry_base = number("dry_base", opt.dry_base);
        const long long dry_allowed = std::max(0LL, integer("dry_allowed_length", opt.dry_allowed_length));
        std::vector<std::string> breakers = opt.dry_breakers;
        if (body.contains("dry_sequence_breakers") && body.at("dry_sequence_breakers").is_array()) {
            breakers.clear();
            for (const auto& e : body.at("dry_sequence_breakers"))
                if (e.is_string()) breakers.push_back(e.get<std::string>());
        }
        if (breakers.empty()) breakers = {"\n", ":", "\"", "*"};
        std::string enc;
        for (const std::string& s : breakers) {
            if (s.empty()) continue;
            const std::vector<llama_token> ids = common_tokenize(vocab, s, /*add_special=*/false, /*parse_special=*/true);
            if (ids.empty()) continue;
            if (!enc.empty()) enc += '|';
            for (size_t k = 0; k < ids.size(); ++k) { if (k) enc += ','; enc += std::to_string((long long) ids[k]); }
        }
        keys += " dry_multiplier=" + text(dry_multiplier) + " dry_base=" + text(dry_base) +
                " dry_allowed_length=" + std::to_string(dry_allowed) +
                " dry_penalty_last_n=" + std::to_string(dry_window);
        if (!enc.empty()) keys += " dry_breakers=" + enc;
    }
    return keys;
}

/// The parse of one response as its text grows: what has been published, and what to publish next.
///
/// Reasoning and answer text are published as they arrive.  Tool calls are published once, from the parse of the
/// complete message: while a call is being written its arguments are partial JSON that the parser closes and
/// normalizes on every token, so text already parsed can change (an escape, a number still being written), and
/// an OpenAI stream cannot take back what it sent.  The same rule keeps the answer text sound: a partial parse
/// whose content does not extend what was published (the parser reread the tail as the start of a tool call)
/// publishes nothing for the content until it does.  This changes delivery, never generated tokens.
struct StreamParse {
    common_chat_msg msg;                // what has been published
    std::vector<std::string> call_ids;

    std::vector<common_chat_msg_diff> step(const std::string& text, bool partial, const common_chat_parser_params& parser) {
        common_chat_msg now;
        try {
            now = common_chat_parse(text, partial, parser);
        } catch (const std::exception&) {
            if (partial) return {};                             // not parseable yet: wait for more text
            now = common_chat_msg();
            now.role = "assistant";
            now.content = text;
        }
        now.content_parts.clear();
        if (partial) {
            now.tool_calls.clear();
            if (now.content.compare(0, msg.content.size(), msg.content) != 0) now.content = msg.content;
            if (now.reasoning_content.compare(0, msg.reasoning_content.size(), msg.reasoning_content) != 0)
                now.reasoning_content = msg.reasoning_content;
        } else {
            // the complete parse decides; text it no longer holds was already sent and stays sent (the parser
            // trims the white space before a tool call, which the partial parses still carried)
            for (auto field : {&common_chat_msg::content, &common_chat_msg::reasoning_content}) {
                const std::string& sent = msg.*field;
                std::string& whole = now.*field;
                if (whole.compare(0, sent.size(), sent) == 0) continue;
                if (sent.compare(0, whole.size(), whole) != 0)
                    std::fprintf(stderr, "unbiased-strata: the complete message's %s differs from the %zu bytes "
                                         "already streamed; the stream keeps what it sent\n",
                                 field == &common_chat_msg::content ? "content" : "reasoning", sent.size());
                whole = sent;
            }
        }
        if (now.empty()) return {};
        now.set_tool_call_ids(call_ids, [] { return "call_" + make_id().substr(9); });
        std::vector<common_chat_msg_diff> diffs = common_chat_msg_diff::compute_diffs(msg, now);
        msg = now;
        return diffs;
    }
};

struct Server {
    Options opt;
    Engine engine;
    strata_frontend::Vision vision;
    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;
    common_chat_templates_ptr tmpls;

    /// Parses an OpenAI chat request, renders the model's chat template and tokenizes the result.
    bool prepare(const common_json& body, Prepared& p, std::string& err) {
        if (!body.is_object() || !body.contains("messages")) { err = "the request needs a \"messages\" array"; return false; }
        common_chat_templates_inputs in;
        common_json messages = body.at("messages");
        std::vector<std::string> markers;
        std::vector<std::shared_ptr<strata_frontend::EncodedImage>> images;
        {
            std::lock_guard<std::mutex> lock(engine.mu);
            for (auto& message : messages) {
                if (!message.contains("content") || !message.at("content").is_array()) continue;
                for (auto& part : message["content"]) {
                    if (part.value("type", "") != "image_url") continue;
                    if (images.size() >= 32) throw std::invalid_argument("at most 32 images per request");
                    const std::string url = part.at("image_url").at("url").get<std::string>();
                    images.push_back(vision.encode(url));
                    markers.push_back("UNBIASED_IMAGE_" + make_id() + "_" + std::to_string(markers.size()));
                    part = common_json::object();
                    part["type"] = "media_marker"; part["text"] = markers.back();
                }
            }
        }
        in.messages = common_chat_msgs_parse_oaicompat(messages);
        if (body.contains("tools") && body.at("tools").is_array() && body.at("tools").size() > 0) {
            in.tools = common_chat_tools_parse_oaicompat(body.at("tools"));
            if (body.contains("tool_choice") && body.at("tool_choice").is_string())
                in.tool_choice = common_chat_tool_choice_parse_oaicompat(body.at("tool_choice").get<std::string>());
            const auto caps = common_chat_templates_get_caps(tmpls.get());
            in.parallel_tool_calls = body.value("parallel_tool_calls", caps.at("supports_parallel_tool_calls"));
        }
        in.add_generation_prompt = true;
        in.use_jinja = true;
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;   // reasoning goes to "reasoning_content"
        in.enable_thinking = true;
        // thinking: OpenAI's reasoning_effort, or the llama.cpp / vLLM convention chat_template_kwargs.enable_thinking
        if (body.contains("reasoning_effort") && body.at("reasoning_effort").is_string()) {
            const std::string effort = body.at("reasoning_effort").get<std::string>();
            if (effort == "none") in.enable_thinking = false;
            else in.chat_template_kwargs["reasoning_effort"] = common_json(effort).dump();
        }
        if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
            const common_json& kw = body.at("chat_template_kwargs");
            if (kw.contains("enable_thinking") && kw.at("enable_thinking").is_boolean())
                in.enable_thinking = kw.at("enable_thinking").get<bool>();
            if (kw.contains("reasoning_effort") && kw.at("reasoning_effort").is_string())
                in.chat_template_kwargs["reasoning_effort"] = kw.at("reasoning_effort").dump();
            // Qwen3.6's template drops the reasoning of every reply before the last user message unless this is
            // set; with it (and the client sending reasoning_content back) the history renders as it was
            // generated, so the next turn reuses the checkpoint at the end of the reply, not only the one at
            // its header
            if (kw.contains("preserve_thinking") && kw.at("preserve_thinking").is_boolean())
                in.chat_template_kwargs["preserve_thinking"] = kw.at("preserve_thinking").dump();
        }
        // #123: a hard thinking cap.  The request's own reasoning_budget_tokens / reasoning_budget_message win; the
        // command line sets the default for every request.  A bad value is a 400 before anything is generated.
        p.thinking = in.enable_thinking;
        p.reasoning_budget = opt.reasoning_budget;
        p.reasoning_budget_message = opt.reasoning_budget_message;
        if (body.contains("reasoning_budget_tokens")) {
            const common_json& v = body.at("reasoning_budget_tokens");
            if (!v.is_number_integer() || v.get<long long>() < 0) {
                err = "reasoning_budget_tokens must be 0 or more (0: no budget)";
                return false;
            }
            p.reasoning_budget = v.get<long long>();
        }
        if (body.contains("reasoning_budget_message")) {
            if (!body.at("reasoning_budget_message").is_string()) {
                err = "reasoning_budget_message must be a string";
                return false;
            }
            p.reasoning_budget_message = body.at("reasoning_budget_message").get<std::string>();
        }
        if (!p.thinking) p.reasoning_budget = 0;   // thinking off: nothing to cap
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
        p.thinking_end_tags = cp.thinking_end_tags;
        std::string rendered = cp.prompt;
        if (images.empty()) {
            p.ids = common_tokenize(vocab, rendered, false, true);
        } else {
            const auto pad = common_tokenize(vocab, "<|image_pad|>", false, true);
            const auto literal_pad = common_tokenize(vocab, "<|image_pad|>", false, false);
            const auto start = common_tokenize(vocab, "<|vision_start|>", false, true);
            const auto end = common_tokenize(vocab, "<|vision_end|>", false, true);
            if (pad.size() != 1 || start.size() != 1 || end.size() != 1)
                throw std::runtime_error("model does not have the expected image special tokens");
            auto text_ids = [&](const std::string& text) {
                for (auto id : common_tokenize(vocab, text, false, true)) {
                    if (id == pad[0]) p.ids.insert(p.ids.end(), literal_pad.begin(), literal_pad.end());
                    else p.ids.push_back(id);
                }
            };
            size_t offset = 0; rendered.clear();
            for (size_t i = 0; i < images.size(); ++i) {
                auto at = cp.prompt.find(markers[i], offset);
                if (at == std::string::npos || cp.prompt.find(markers[i], at+markers[i].size()) != std::string::npos)
                    throw std::invalid_argument("chat template did not preserve each image exactly once");
                auto text = cp.prompt.substr(offset, at-offset);
                text_ids(text); rendered += text + "<|vision_start|>";
                p.ids.push_back(start[0]);
                p.ids.insert(p.ids.end(), images[i]->tokens, pad[0]);
                for (int j = 0; j < images[i]->tokens; ++j) rendered += "<|image_pad|>";
                p.ids.push_back(end[0]); rendered += "<|vision_end|>";
                offset = at + markers[i].size();
            }
            auto tail = cp.prompt.substr(offset); text_ids(tail); rendered += tail;
            p.images = std::make_shared<strata_frontend::ImageFile>();
            strata_frontend::write_images(*p.images, images);
        }
        if (p.ids.empty()) { err = "the prompt is empty"; return false; }

        p.parser = common_chat_parser_params(cp);
        p.parser.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        p.parser.parse_tool_calls = true;
        if (!cp.parser.empty()) p.parser.parser.load(cp.parser);

        // The engine refuses a request unless prompt + max_new + 8 <= context (generate.cpp, the GEN handler), so
        // the slots it keeps back are not ours to promise.  Without this a client that asks for "as much as fits"
        // (pi sends max_tokens 32768 on a 64K card) was refused on every prompt longer than ctx - max_tokens.
        constexpr long long kEngineReserve = 8;
        const long long n = (long long) p.ids.size();
        const long long room = opt.ctx - n - kEngineReserve;
        if (room < 1) {
            err = "the prompt is " + std::to_string(n) + " tokens, which leaves no room to generate in the context of " +
                  std::to_string(opt.ctx);
            return false;
        }
        long long max_new = room;
        for (const char* key : {"max_completion_tokens", "max_tokens"})
            if (body.contains(key) && body.at(key).is_number()) {
                const long long want = body.at(key).get<long long>();
                if (want >= 1) {
                    if (want <= room) max_new = want;
                    else if (opt.fit_max_tokens) max_new = room;
                    else {
                        err = "the request asks for " + std::to_string(want) + " tokens and the prompt is " +
                              std::to_string(n) + ", which does not fit the context of " + std::to_string(opt.ctx) +
                              " (--fit-max-tokens shortens the request to the room left)";
                        return false;
                    }
                }
                break;
            }
        p.max_new = max_new;
        p.stream = body.value("stream", false);

        // sampling: the launch-time defaults with the request's own fields winning (see sampling_keys)
        const std::string keys = sampling_keys(body, opt, vocab);
        // The header's parts are kept so a continuation (a thinking budget wrap-up) can rebuild it with a smaller
        // max_new; the assembled header is byte-for-byte the one this fork always sent.
        p.gen_prefix = p.images ? "GENI " : "GEN ";
        p.gen_suffix = keys + " " + (p.images ? p.images->path + " " : "");
        p.header = p.gen_prefix + std::to_string(max_new) + p.gen_suffix;
        // Opt-in provenance capture: the exact first rendered request this
        // process serves, including token IDs and the engine sampling header.
        if (const char* path = std::getenv("SUGMEOFF_CAPTURE_FIRST_PROMPT")) {
            if (FILE* f = std::fopen(path, "wx")) {
                common_json capture = common_json::object();
                capture["request"] = body;
                capture["rendered_prompt"] = rendered;
                capture["token_ids"] = p.ids;
                capture["engine_header"] = p.header;
                const std::string text = capture.dump_safe();
                if (std::fwrite(text.data(), 1, text.size(), f) != text.size())
                    std::fprintf(stderr, "unbiased-strata: prompt capture write failed: %s\n", path);
                std::fclose(f);
            }
        }
        return true;
    }

    common_json usage_json(const GenResult& r) const {
        common_json u = common_json::object();
        u["prompt_tokens"] = r.prompt_n;
        u["completion_tokens"] = r.produced;
        u["total_tokens"] = r.prompt_n + r.produced;
        u["completion_tokens_details"] = common_json::object();
        u["completion_tokens_details"]["reasoning_tokens"] = r.reasoning_tokens;
        return u;
    }

    common_json timings_json(const GenResult& r) const {
        common_json t = common_json::object();
        t["prompt_n"] = r.prompt_n;
        t["prompt_ms"] = r.prompt_ms;
        t["prompt_per_second"] = r.prompt_ms > 0 ? 1000.0 * (double) r.prompt_n / r.prompt_ms : 0.0;
        t["predicted_n"] = r.produced;
        t["predicted_ms"] = r.decode_ms;
        t["predicted_per_second"] = r.decode_ms > 0 ? 1000.0 * (double) r.produced / r.decode_ms : 0.0;
        t["draft_n"] = r.draft_offered;
        t["draft_n_accepted"] = r.draft_accepted;
        return t;
    }

    /// Runs one request.  The text produced so far is parsed by llama.cpp's chat parser after every token, and
    /// `emit` gets what changed (reasoning and answer text as they grow, tool calls once the message is complete:
    /// see StreamParse); it returns false when the client has gone.  `final_msg` is the whole parsed message.
    /// `keepalive`, when given, is called whenever 15 s pass without an emit (a long tool call is silent while
    /// its arguments are written): the stream writes an SSE comment, which keeps clients with an idle timeout
    /// (pi: 5 minutes) attached and notices a client that has gone, so the generation is cancelled instead of
    /// running to the end for nobody.
    GenResult run(const Prepared& p, common_chat_msg& final_msg,
                  const std::function<bool(const std::vector<common_chat_msg_diff>&)>& emit,
                  const std::function<bool()>& keepalive = {}) {
        std::lock_guard<std::mutex> lk(engine.mu);
        std::fprintf(stderr, "unbiased-strata: request %s start: prompt=%zu header=%s\n",
                     p.request_id.c_str(), p.ids.size(), p.header.c_str());
        std::string text, pending;
        StreamParse parse;
        common_chat_msg& msg = parse.msg;
        bool client_ok = true;
        auto last_write = std::chrono::steady_clock::now();
        auto update = [&](bool partial) {
            const std::vector<common_chat_msg_diff> diffs = parse.step(text, partial, p.parser);
            if (client_ok && !diffs.empty()) {
                client_ok = emit(diffs);
                last_write = std::chrono::steady_clock::now();
            }
        };
        // #123: a hard thinking cap (reasoning_budget_tokens).  When the thinking reaches the budget it is closed
        // with the budget message and the model's end tag, and the model answers from there.  The wrap-up is fed
        // like generated text and the engine is asked again with prompt + generated + wrap-up, continuing from the
        // prefix it already holds, so nothing is read twice.  Off unless thinking is on and a budget is set.
        bool budget_on = p.thinking && p.reasoning_budget > 0;
        long long thought = 0;              // reasoning tokens generated so far (the budget's count)
        long long produced = 0;             // every generated token, the wrap-up included
        bool seen_end = false;              // an end-of-thinking tag is in `text`
        auto reasoning_over = [&]() {
            for (const std::string& tag : p.thinking_end_tags)
                if (!tag.empty() && text.find(tag) != std::string::npos) return true;
            return false;
        };
        auto feed = [&](llama_token id) {
            pending += common_token_to_piece(vocab, id, /*special=*/true);
            const size_t n = utf8_complete_prefix(pending);
            if (n > 0) {
                text.append(pending, 0, n);
                pending.erase(0, n);
                if (p.stream) update(true);                     // a whole response is parsed once, at the end
            }
        };
        GenResult r;
        r.ok = true;
        std::vector<llama_token> prompt_ids = p.ids;
        while (true) {
            bool budget_reached = false;
            std::vector<llama_token> pass_ids;
            const std::string header = p.gen_prefix + std::to_string(p.max_new - produced) + p.gen_suffix;
            GenResult pass = engine.generate(header, prompt_ids, [&](llama_token id) {
                if (llama_vocab_is_eog(vocab, id)) return client_ok;
                pass_ids.push_back(id);
                if (budget_on && !seen_end) ++thought;
                feed(id);
                if (!seen_end && reasoning_over()) seen_end = true;
                if (budget_on && !seen_end && thought >= p.reasoning_budget && pending.empty()) {
                    budget_reached = true;                  // engine STOP; the client is still here
                    return false;
                }
                if (client_ok && keepalive) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now - last_write >= std::chrono::seconds(15)) {
                        client_ok = keepalive();
                        last_write = now;
                    }
                }
                return client_ok;
            });
            if (!pass.ok) { r = pass; break; }
            produced += pass.produced;
            r.ok = true;
            r.finish = pass.finish;
            r.prompt_ms += pass.prompt_ms;
            r.decode_ms += pass.decode_ms;
            r.draft_accepted += pass.draft_accepted;
            r.draft_offered += pass.draft_offered;
            if (!budget_reached || !client_ok) break;
            budget_on = false;                  // wrap the thinking up at most once
            const std::string end_tag = p.thinking_end_tags.empty() ? std::string(" response")
                                                                   : p.thinking_end_tags.front();
            const std::vector<llama_token> extra =
                common_tokenize(vocab, "\n\n" + p.reasoning_budget_message + end_tag + "\n\n", false, true);
            if (p.max_new - produced - (long long) extra.size() < 1) break;   // no room to answer: "length"
            std::fprintf(stderr, "unbiased-strata: request %s thinking budget reached (%lld tokens): wrapping up\n",
                         p.request_id.c_str(), thought);
            for (llama_token t : extra) { pass_ids.push_back(t); feed(t); }
            prompt_ids.insert(prompt_ids.end(), pass_ids.begin(), pass_ids.end());
            produced += (long long) extra.size();
        }
        if (r.ok) {
            r.prompt_n = (long long) p.ids.size();   // usage reports the request's own prompt, not a continuation's
            r.produced = produced;
            r.reasoning_tokens = thought;
        }
        text += pending;
        try {
            update(false);
        } catch (const std::exception& ex) {
            r.ok = false;
            r.error = std::string("response formatter failed: ") + ex.what();
            std::fprintf(stderr, "unbiased-strata: final response formatter failed: %s\n", ex.what());
        }
        final_msg = msg;
        if (final_msg.role.empty()) final_msg.role = "assistant";
        if (final_msg.tool_calls.empty() && text.find("<tool_call>") != std::string::npos) {
            // The parser accepts a message with no tool call even when the text holds one it could not
            // match (see third_party/llama.cpp/common/parsers/qwen3-coder.cpp); say so, with the text.
            std::fprintf(stderr, "unbiased-strata: request %s: the output contains <tool_call> but no tool call was "
                                 "parsed; the client sees finish=stop. raw-output-begin\n%s\n"
                                 "unbiased-strata: request %s raw-output-end\n",
                         p.request_id.c_str(), text.c_str(), p.request_id.c_str());
        }
        if (r.error.find("response formatter failed") == 0) {
            std::fprintf(stderr, "unbiased-strata: request %s formatter-input-begin\n%s\n"
                                 "unbiased-strata: request %s formatter-input-end\n",
                         p.request_id.c_str(), text.c_str(), p.request_id.c_str());
        }
        std::fprintf(stderr, "unbiased-strata: request %s end: ok=%d prompt=%lld generated=%lld finish=%s error=%s\n",
                     p.request_id.c_str(), r.ok ? 1 : 0, r.prompt_n, r.produced, r.finish.c_str(), r.error.c_str());
        return r;
    }

    static const char* finish_reason(const GenResult& r, const common_chat_msg& msg) {
        if (!msg.tool_calls.empty()) return "tool_calls";
        return r.finish == "stop" ? "stop" : "length";
    }

    void handle_chat(const httplib::Request& req, httplib::Response& res) {
        if (!engine.alive()) return json_error(res, 503, "the engine is not running", "server_error");
        const common_json body = common_json::parse_no_throw(req.body);
        if (body.is_discarded()) return json_error(res, 400, "the request body is not valid JSON", "invalid_request_error");
        auto prep = std::make_shared<Prepared>();
        std::string err;
        try {
            if (!prepare(body, *prep, err)) return json_error(res, 400, err, "invalid_request_error");
        } catch (const std::exception& ex) {
            return json_error(res, 400, ex.what(), "invalid_request_error");
        }
        const std::string id = make_id();
        prep->request_id = id;
        const long long created = now_s();

        if (!prep->stream) {
            common_chat_msg msg;
            const GenResult r = run(*prep, msg, [](const std::vector<common_chat_msg_diff>&) { return true; });
            if (!r.ok) return json_error(res, 500, r.error, "server_error");
            common_json choice = common_json::object();
            choice["index"] = 0;
            choice["message"] = msg.to_json_oaicompat();
            choice["finish_reason"] = finish_reason(r, msg);
            common_json out = common_json::object();
            out["id"] = id;
            out["object"] = "chat.completion";
            out["created"] = created;
            out["model"] = g_model_name;
            out["choices"] = common_json::array();
            out["choices"].push_back(choice);
            out["usage"] = usage_json(r);
            out["timings"] = timings_json(r);
            res.set_content(out.dump_safe(), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [this, prep, id, created](size_t, httplib::DataSink& sink) {
            auto chunk = [&](const common_json& delta, const char* finish, const GenResult* done) {
                common_json choice = common_json::object();
                choice["index"] = 0;
                choice["delta"] = delta;
                if (finish) choice["finish_reason"] = finish; else choice["finish_reason"] = nullptr;
                common_json out = common_json::object();
                out["id"] = id;
                out["object"] = "chat.completion.chunk";
                out["created"] = created;
                out["model"] = g_model_name;
                out["choices"] = common_json::array();
                out["choices"].push_back(choice);
                if (done) { out["usage"] = usage_json(*done); out["timings"] = timings_json(*done); }
                const std::string s = "data: " + out.dump_safe() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            common_json first = common_json::object();
            first["role"] = "assistant";
            first["content"] = "";
            bool ok = chunk(first, nullptr, nullptr);
            common_chat_msg msg;
            const GenResult r = run(*prep, msg, [&](const std::vector<common_chat_msg_diff>& diffs) {
                for (const common_chat_msg_diff& df : diffs) {
                    common_json d = common_json::object();
                    if (!df.reasoning_content_delta.empty()) d["reasoning_content"] = df.reasoning_content_delta;
                    if (!df.content_delta.empty()) d["content"] = df.content_delta;
                    if (df.tool_call_index != std::string::npos) {
                        common_json fn = common_json::object();
                        if (!df.tool_call_delta.name.empty()) fn["name"] = df.tool_call_delta.name;
                        fn["arguments"] = df.tool_call_delta.arguments;
                        common_json tc = common_json::object();
                        tc["index"] = (long long) df.tool_call_index;
                        if (!df.tool_call_delta.id.empty()) { tc["id"] = df.tool_call_delta.id; tc["type"] = "function"; }
                        tc["function"] = fn;
                        d["tool_calls"] = common_json::array();
                        d["tool_calls"].push_back(tc);
                    }
                    if (d.size() == 0) continue;
                    ok = ok && chunk(d, nullptr, nullptr);
                }
                return ok;
            }, [&]() {
                static const std::string ka = ": keepalive\n\n";   // an SSE comment: every client ignores it
                ok = ok && sink.write(ka.data(), ka.size());
                return ok;
            });
            if (ok) {
                if (r.ok) {
                    chunk(common_json::object(), finish_reason(r, msg), &r);
                } else {
                    common_json e = common_json::object();
                    e["error"] = common_json::object();
                    e["error"]["message"] = r.error;
                    e["error"]["type"] = "server_error";
                    const std::string s = "data: " + e.dump_safe() + "\n\n";
                    sink.write(s.data(), s.size());
                }
                static const std::string end = "data: [DONE]\n\n";
                sink.write(end.data(), end.size());
            }
            sink.done();
            return true;
        });
    }
};

Server* g_server = nullptr;
httplib::Server* g_http = nullptr;

void on_signal(int) {
    if (g_http) g_http->stop();
}

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// The CPU expert workers to ask the engine for when --threads is not given: 0 lets the engine use every physical
// core it sees.  Inside a container it sees the host's cores, not its own share, and that many workers on a small
// quota throttle one another; so with a cgroup CPU quota below the visible CPUs the quota decides.  The quota
// counts hardware threads and the engine wants cores, hence the halving when SMT is on.
int default_threads() {
    long long quota = 0, period = 0;
    char q[32] = "";
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {                       // cgroup v2
        if (std::fscanf(f, "%31s %lld", q, &period) == 2 && std::strcmp(q, "max") != 0) quota = std::atoll(q);
        std::fclose(f);
    } else if (FILE* f1 = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {  // cgroup v1
        if (std::fscanf(f1, "%lld", &quota) != 1) quota = 0;
        std::fclose(f1);
        if (FILE* f2 = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(f2, "%lld", &period) != 1) period = 0;
            std::fclose(f2);
        }
    }
    if (quota <= 0 || period <= 0) return 0;
    const long long cpus = (quota + period / 2) / period;
    if (cpus < 1 || cpus >= sysconf(_SC_NPROCESSORS_ONLN)) return 0;
    int smt = 0;
    if (FILE* f = std::fopen("/sys/devices/system/cpu/smt/active", "r")) {
        if (std::fscanf(f, "%d", &smt) != 1) smt = 0;
        std::fclose(f);
    }
    return (int) std::max<long long>(smt ? cpus / 2 : cpus, 1);
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "unbiased-strata: %s needs a value\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model") o.model = next("--model");
        else if (a == "--mmproj") o.mmproj = next("--mmproj");
        else if (a == "--vision-tokens") o.vision_tokens = std::atoi(next("--vision-tokens").c_str());
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--host") o.host = next("--host");
        else if (a == "--port") o.port = std::atoi(next("--port").c_str());
        else if (a == "--ctx") o.ctx = std::atoll(next("--ctx").c_str());
        else if (a == "--fit-max-tokens") o.fit_max_tokens = true;
        else if (a == "--no-fit-max-tokens") o.fit_max_tokens = false;
        else if (a == "--reasoning-budget-tokens") o.reasoning_budget = std::atoll(next("--reasoning-budget-tokens").c_str());
        else if (a == "--reasoning-budget-message") o.reasoning_budget_message = next("--reasoning-budget-message");
        else if (a == "--threads") o.threads = std::atoi(next("--threads").c_str());
        else if (a == "--vram-cap") o.vram_cap_mib = std::atoll(next("--vram-cap").c_str());
        else if (a == "--ram-cap") o.ram_cap_gib = std::atof(next("--ram-cap").c_str());
        else if (a == "--temperature" || a == "--temp") o.temperature = std::atof(next("--temperature").c_str());
        else if (a == "--top-k") o.top_k = std::atoll(next("--top-k").c_str());
        else if (a == "--top-p") o.top_p = std::atof(next("--top-p").c_str());
        else if (a == "--min-p") o.min_p = std::atof(next("--min-p").c_str());
        else if (a == "--repeat-last-n") o.repeat_last_n = std::atoll(next("--repeat-last-n").c_str());
        else if (a == "--repeat-penalty") o.repeat_penalty = std::atof(next("--repeat-penalty").c_str());
        else if (a == "--presence-penalty") o.presence_penalty = std::atof(next("--presence-penalty").c_str());
        else if (a == "--frequency-penalty") o.frequency_penalty = std::atof(next("--frequency-penalty").c_str());
        else if (a == "--dry-multiplier") o.dry_multiplier = std::atof(next("--dry-multiplier").c_str());
        else if (a == "--dry-base") o.dry_base = std::atof(next("--dry-base").c_str());
        else if (a == "--dry-allowed-length") o.dry_allowed_length = std::atoll(next("--dry-allowed-length").c_str());
        else if (a == "--dry-penalty-last-n") o.dry_penalty_last_n = std::atoll(next("--dry-penalty-last-n").c_str());
        else if (a == "--dry-breakers") {
            const std::string list = next("--dry-breakers");
            o.dry_breakers.clear();
            for (size_t p = 0; p <= list.size();) {
                const size_t bar = list.find('|', p);
                const size_t end = bar == std::string::npos ? list.size() : bar;
                o.dry_breakers.push_back(list.substr(p, end - p));
                if (bar == std::string::npos) break;
                p = bar + 1;
            }
        }
        else if (a == "--seed") o.seed = std::atoll(next("--seed").c_str());
        else if (a == "--") {
            for (++i; i < argc; ++i) {
                const std::string e = argv[i];
                // one GPU only: the engine's multi-GPU and remote-expert modes are not part of this fork
                for (const char* no : {"--gpus", "--layer-split", "--split-device", "--peer-device", "--peer-slots",
                                       "--remote-expert", "--vision", "--batch"})
                    if (e.rfind(no, 0) == 0) {
                        std::fprintf(stderr, "unbiased-strata: %s is not supported here (use --mmproj for image input)\n", e.c_str());
                        std::exit(2);
                    }
                o.extra.push_back(e);
            }
        }
        else if (a == "-h" || a == "--help") return false;
        else { std::fprintf(stderr, "unbiased-strata: unknown option %s\n\n", a.c_str()); return false; }
    }
    if (o.reasoning_budget < 0) {
        std::fprintf(stderr, "unbiased-strata: --reasoning-budget-tokens must be 0 or more (0: no budget)\n\n");
        return false;
    }
    if (!std::isfinite(o.temperature) || !std::isfinite(o.top_p) || !std::isfinite(o.min_p) ||
        !std::isfinite(o.repeat_penalty) || !std::isfinite(o.presence_penalty) || !std::isfinite(o.frequency_penalty) ||
        !std::isfinite(o.dry_multiplier) || !std::isfinite(o.dry_base) ||
        o.repeat_penalty <= 0.0 || o.top_p <= 0.0 || o.top_p > 1.0 || o.min_p < 0.0 || o.min_p > 1.0 ||
        o.repeat_last_n < 0 || o.dry_base < 1.0 || o.dry_allowed_length < 0 || o.dry_penalty_last_n < 0) {
        std::fprintf(stderr, "unbiased-strata: sampling options must be finite (--repeat-penalty positive, "
                             "--top-p/--min-p in 0..1, --repeat-last-n 0 or more, --dry-base >= 1, "
                             "--dry-allowed-length/--dry-penalty-last-n 0 or more)\n\n");
        return false;
    }
    return !o.model.empty() && !o.pack.empty() && !o.mtp.empty();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--engine") == 0) return strata_engine_main(argc - 1, argv + 1);

    Server srv;
    Options& o = srv.opt;
    if (!parse_args(argc, argv, o)) { usage(); return 2; }
    g_model_name = model_name_of(o.model);

    // every file the engine needs must already be here: nothing is fetched
    // Flash-Next ships as two shards and the second holds the PLE table, so it is the engine's --ple-gguf; a one-file
    // model (Qwen3.6-35B-A3B) has no PLE and no second shard, and the engine decides from the file's architecture
    std::string shard2 = o.model;
    const size_t at = shard2.rfind("-00001-of-00002");
    if (at != std::string::npos) shard2.replace(at, 15, "-00002-of-00002");
    else shard2.clear();
    const std::string profile = o.pack + "/expert-profile.bin";
    // draft_vocab.bin is the draft head's vocabulary subset; a Qwen3.6 draft pack built with the full head
    // (tools/mtp_qwen36_pack.py --full-head) has none and the engine then drafts over the whole vocabulary
    for (const std::string& f : {o.model, shard2, o.pack + "/index.txt", profile, o.mtp + "/experts.bin"})
        if (!f.empty() && !file_exists(f)) { std::fprintf(stderr, "unbiased-strata: missing file: %s\n", f.c_str()); return 2; }
    if (o.ctx < 1024) { std::fprintf(stderr, "unbiased-strata: --ctx must be at least 1024\n"); return 2; }

    if (!o.mmproj.empty() && (!file_exists(o.mmproj) || o.vision_tokens < 16 || o.vision_tokens > 4096)) {
        std::fprintf(stderr, "unbiased-strata: --mmproj must exist and --vision-tokens must be 16..4096\n"); return 2;
    }
    // the engine, started first: loading takes minutes, and it must be forked before any thread exists
    std::vector<std::string> eargs = {
        "unbiased-strata", "--engine", "--serve", "--pack", o.pack, "--native", o.model,
        "--expert-profile", profile, "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
        "--mtp", o.mtp, "--max-context", std::to_string(o.ctx)};
    if (!shard2.empty()) { eargs.push_back("--ple-gguf"); eargs.push_back(shard2); eargs.push_back("--ple-io"); eargs.push_back("ram"); }
    if (o.vram_cap_mib > 0) { eargs.push_back("--vram-cap"); eargs.push_back(std::to_string(o.vram_cap_mib)); }
    if (o.ram_cap_gib > 0.0) { eargs.push_back("--ram-cap"); eargs.push_back(std::to_string(o.ram_cap_gib)); }
    if (!o.mmproj.empty()) eargs.push_back("--vision");
    if (o.ctx > 8192) { eargs.push_back("--kv"); eargs.push_back("int8"); }
    if (o.threads <= 0) {
        o.threads = default_threads();
        if (o.threads > 0)
            std::fprintf(stderr, "unbiased-strata: this container's CPU quota gives %d expert threads (--threads N overrides)\n",
                         o.threads);
    }
    if (o.threads > 0) { eargs.push_back("--pool-workers"); eargs.push_back(std::to_string(o.threads)); }
    for (const std::string& e : o.extra) eargs.push_back(e);
    std::signal(SIGPIPE, SIG_IGN);
    std::string err;
    if (!srv.engine.start(eargs, err)) { std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str()); return 1; }

    // the tokenizer and the chat template, from the GGUF itself
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    srv.model = llama_model_load_from_file(o.model.c_str(), mp);
    if (srv.model == nullptr) {
        std::fprintf(stderr, "unbiased-strata: could not read the tokenizer from %s\n", o.model.c_str());
        srv.engine.quit();
        return 1;
    }
    srv.vocab = llama_model_get_vocab(srv.model);
    srv.tmpls = common_chat_templates_init(srv.model, "");
    if (!o.mmproj.empty()) {
        try { srv.vision.init(o.mmproj, srv.model, o.threads, o.vision_tokens); }
        catch (const std::exception& ex) {
            std::fprintf(stderr, "unbiased-strata: %s\n", ex.what()); srv.engine.quit(); return 1;
        }
    }

    std::fprintf(stderr, "unbiased-strata: loading the model (this takes a few minutes)...\n");
    long long engine_ctx = 0;
    if (!srv.engine.wait_ready(err, &engine_ctx)) {
        std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str());
        srv.engine.quit();
        return 1;
    }
    if (engine_ctx > 0 && engine_ctx != o.ctx) {   // the engine's VRAM plan lowered the context: serve that one
        std::fprintf(stderr, "unbiased-strata: the engine allocated a context of %lld tokens, not the %lld requested; "
                             "requests are bounded by %lld\n", engine_ctx, o.ctx, engine_ctx);
        o.ctx = engine_ctx;
    }

    httplib::Server http;
    g_server = &srv;
    g_http = &http;
    // /health: ok or error, plus the engine's expert tiers and counters when it is between requests (STATS: VRAM
    // slots and how many hold an expert, RAM copies, mapped experts, tokens served, VRAM hits of lookups, RAM-copy
    // and file reads, warm hints, major page faults).  During a request the last reading is served, marked stale.
    // httplib serves /health from several threads: the last reading has its own lock (engine.mu is the engine's,
    // taken only to ask), and every read of it is a copy under that lock.
    static std::mutex stats_mu;
    static std::string last_stats;
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = srv.engine.alive() ? 200 : 503;
        std::string body = srv.engine.alive() ? "{\"status\":\"ok\"" : "{\"status\":\"error\"";
        if (srv.engine.alive()) {
            std::unique_lock<std::mutex> lk(srv.engine.mu, std::try_to_lock);
            if (lk.owns_lock()) {
                std::string reply, err;
                const bool got = srv.engine.command("STATS", "STATS ", reply, err);
                lk.unlock();
                std::string stats;
                {
                    std::lock_guard<std::mutex> sl(stats_mu);
                    if (got) last_stats = reply.substr(6);
                    stats = last_stats;
                }
                if (!got && stats.empty()) body += ",\"tiers_error\":\"" + err + "\"";
                if (!stats.empty()) body += ",\"tiers\":" + stats;
            } else {
                std::string stats;
                {
                    std::lock_guard<std::mutex> sl(stats_mu);
                    stats = last_stats;
                }
                if (!stats.empty()) body += ",\"tiers\":" + stats + ",\"tiers_stale\":true";
                else body += ",\"busy\":true";
            }
        }
        body += "}";
        res.set_content(body, "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        if (!srv.engine.alive()) return json_error(res, 503, "the engine is not running", "server_error");
        common_json m = common_json::object();
        m["id"] = g_model_name;
        m["object"] = "model";
        m["owned_by"] = "local";
        m["context_length"] = srv.opt.ctx;
        m["input_modalities"] = common_json::array();
        m["input_modalities"].push_back("text");
        if (srv.vision.enabled()) m["input_modalities"].push_back("image");
        common_json out = common_json::object();
        out["object"] = "list";
        out["data"] = common_json::array();
        out["data"].push_back(m);
        res.set_content(out.dump_safe(), "application/json");
    });
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        srv.handle_chat(req, res);
    });
    http.set_payload_max_length(64 * 1024 * 1024);
    http.set_read_timeout(600, 0);
    http.set_write_timeout(600, 0);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (!http.bind_to_port(o.host, o.port)) {
        std::fprintf(stderr, "unbiased-strata: could not listen on %s:%d\n", o.host.c_str(), o.port);
        srv.engine.quit();
        return 1;
    }
    std::fprintf(stderr, "unbiased-strata: ready on http://%s:%d (context %lld tokens)\n", o.host.c_str(), o.port, o.ctx);
    // The engine watchdog intentionally aborts a stuck child. Exit the HTTP
    // parent too so systemd's Restart=on-failure reloads the complete service.
    // Do not fork a replacement from this multithreaded HTTP process.
    std::atomic<bool> shutting_down{false};
    std::thread supervisor([&] {
        while (!shutting_down.load()) {
            std::string why;
            if (srv.engine.reap_exit(why)) {
                std::fprintf(stderr, "unbiased-strata: %s; exiting frontend for service restart\n", why.c_str());
                std::fflush(stderr);
                std::_Exit(1);
            }
            if (!srv.engine.alive()) {
                std::fprintf(stderr, "unbiased-strata: engine protocol failed; exiting frontend for service restart\n");
                std::fflush(stderr);
                std::_Exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });
    http.listen_after_bind();
    shutting_down.store(true);
    supervisor.join();

    std::fprintf(stderr, "unbiased-strata: stopping\n");
    srv.engine.quit();
    llama_model_free(srv.model);
    return 0;
}
