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

const char* const kModelName = "qwen3.8-flash-next-iq3_xxs";

// The engine's serve loop refuses a request unless prompt + max_new + this margin fits --ctx
// (src/program/generate.cpp:6403): the last cells are the engine's own, not ours to hand out.
constexpr long long kEngineReserve = 8;

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
        "  --threads N     CPU threads for the experts the GPU does not hold (default: every physical core,\n"
        "                  or half the CPU quota inside a container that has one)\n"
        "  --host ADDR     address to listen on (default 127.0.0.1)\n"
        "  --port N        port to listen on (default 8080)\n\n"
        "Serves POST /v1/chat/completions (OpenAI style, with streaming), GET /v1/models and GET /health.\n");
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
    std::string request_id;
    common_chat_parser_params parser;   // how to split the output into reasoning, answer and tool calls
    bool stream = false;
    long long max_new = 0;
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
        }
        const common_chat_params cp = common_chat_templates_apply(tmpls.get(), in);
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

        // sampling: greedy unless the request sets a temperature above zero
        std::string keys;
        const double temperature = body.contains("temperature") && body.at("temperature").is_number()
                                       ? body.at("temperature").get<double>() : 0.0;
        if (temperature > 0.0) {
            const double top_p = body.contains("top_p") && body.at("top_p").is_number() ? body.at("top_p").get<double>() : 0.95;
            long long top_k = body.contains("top_k") && body.at("top_k").is_number() ? body.at("top_k").get<long long>() : 20;
            top_k = std::max(1LL, std::min(64LL, top_k));       // the engine's sampler takes 1..64
            unsigned long long seed;
            if (body.contains("seed") && body.at("seed").is_number()) seed = (unsigned long long) body.at("seed").get<long long>();
            else seed = std::random_device{}();
            char b[160];
            std::snprintf(b, sizeof b, " temperature=%g top_p=%g top_k=%lld seed=%llu", temperature, top_p, top_k, seed);
            keys += b;
            if (body.contains("min_p") && body.at("min_p").is_number()) {
                std::snprintf(b, sizeof b, " min_p=%g", body.at("min_p").get<double>());
                keys += b;
            }
        }
        for (const auto& kv : {std::pair<const char*, const char*>{"presence_penalty", "penalty_present"},
                               std::pair<const char*, const char*>{"frequency_penalty", "penalty_freq"},
                               std::pair<const char*, const char*>{"repeat_penalty", "penalty_repeat"}})
            if (body.contains(kv.first) && body.at(kv.first).is_number()) {
                char b[64];
                std::snprintf(b, sizeof b, " %s=%g", kv.second, body.at(kv.first).get<double>());
                keys += b;
                if (keys.find("penalty_last_n") == std::string::npos) keys += " penalty_last_n=64";
            }
        p.header = (p.images ? "GENI " : "GEN ") + std::to_string(max_new) + keys + " " +
                   (p.images ? p.images->path + " " : "");
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
    /// `emit` gets what changed (reasoning, answer text, tool-call pieces); it returns false when the client has
    /// gone.  `final_msg` is the whole parsed message.  `keepalive`, when given, is called whenever 15 s pass
    /// without an emit (content and tool calls are published only once the message is complete, so a long answer
    /// is silent for minutes): the stream writes an SSE comment, which keeps clients with an idle timeout (pi:
    /// 5 minutes) attached and notices a client that has gone, so the generation is cancelled instead of
    /// running to the end for nobody.
    GenResult run(const Prepared& p, common_chat_msg& final_msg,
                  const std::function<bool(const std::vector<common_chat_msg_diff>&)>& emit,
                  const std::function<bool()>& keepalive = {}) {
        std::lock_guard<std::mutex> lk(engine.mu);
        std::fprintf(stderr, "unbiased-strata: request %s start: prompt=%zu header=%s\n",
                     p.request_id.c_str(), p.ids.size(), p.header.c_str());
        std::string text, pending;
        common_chat_msg msg;
        std::vector<std::string> call_ids;
        bool client_ok = true;
        auto last_write = std::chrono::steady_clock::now();
        auto update = [&](bool partial) {
            common_chat_msg now;
            try {
                now = common_chat_parse(text, partial, p.parser);
            } catch (const std::exception&) {
                if (partial) return;                            // not parseable yet: wait for more text
                now = common_chat_msg();
                now.role = "assistant";
                now.content = text;
            }
            // Partial tool JSON is reparsed and normalized as tokens arrive.
            // Escapes and incomplete arguments can change already-parsed text,
            // which cannot be retracted from an OpenAI stream. Publish content
            // and tool calls once, after the full message is parsed; reasoning
            // can still stream. This changes delivery, never generated tokens.
            if (partial) {
                now.content.clear();
                now.content_parts.clear();
                now.tool_calls.clear();
            }
            if (now.empty()) return;
            now.set_tool_call_ids(call_ids, [] { return "call_" + make_id().substr(9); });
            const std::vector<common_chat_msg_diff> diffs = common_chat_msg_diff::compute_diffs(msg, now);
            msg = now;
            if (client_ok && !diffs.empty()) {
                client_ok = emit(diffs);
                last_write = std::chrono::steady_clock::now();
            }
        };
        GenResult r = engine.generate(p.header, p.ids, [&](llama_token id) {
            if (llama_vocab_is_eog(vocab, id)) return client_ok;
            pending += common_token_to_piece(vocab, id, /*special=*/true);
            const size_t n = utf8_complete_prefix(pending);
            if (n > 0) {
                text.append(pending, 0, n);
                pending.erase(0, n);
                update(true);
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
            out["model"] = kModelName;
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
                out["model"] = kModelName;
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
        else if (a == "--threads") o.threads = std::atoi(next("--threads").c_str());
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
    return !o.model.empty() && !o.pack.empty() && !o.mtp.empty();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--engine") == 0) return strata_engine_main(argc - 1, argv + 1);

    Server srv;
    Options& o = srv.opt;
    if (!parse_args(argc, argv, o)) { usage(); return 2; }

    // every file the engine needs must already be here: nothing is fetched
    std::string shard2 = o.model;
    const size_t at = shard2.rfind("-00001-of-00002");
    if (at == std::string::npos) {
        std::fprintf(stderr, "unbiased-strata: --model must be the first shard (...-00001-of-00002.gguf)\n");
        return 2;
    }
    shard2.replace(at, 15, "-00002-of-00002");
    const std::string profile = o.pack + "/expert-profile.bin";
    for (const std::string& f : {o.model, shard2, o.pack + "/index.txt", profile, o.mtp + "/experts.bin",
                                 o.mtp + "/draft_vocab.bin"})
        if (!file_exists(f)) { std::fprintf(stderr, "unbiased-strata: missing file: %s\n", f.c_str()); return 2; }
    if (o.ctx < 1024) { std::fprintf(stderr, "unbiased-strata: --ctx must be at least 1024\n"); return 2; }

    if (!o.mmproj.empty() && (!file_exists(o.mmproj) || o.vision_tokens < 16 || o.vision_tokens > 4096)) {
        std::fprintf(stderr, "unbiased-strata: --mmproj must exist and --vision-tokens must be 16..4096\n"); return 2;
    }
    // the engine, started first: loading takes minutes, and it must be forked before any thread exists
    std::vector<std::string> eargs = {
        "unbiased-strata", "--engine", "--serve", "--pack", o.pack, "--native", o.model, "--ple-gguf", shard2,
        "--expert-profile", profile, "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
        "--mtp", o.mtp, "--max-context", std::to_string(o.ctx), "--ple-io", "ram"};
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
    if (!srv.engine.wait_ready(err)) {
        std::fprintf(stderr, "unbiased-strata: %s\n", err.c_str());
        srv.engine.quit();
        return 1;
    }

    httplib::Server http;
    g_server = &srv;
    g_http = &http;
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.status = srv.engine.alive() ? 200 : 503;
        res.set_content(srv.engine.alive() ? "{\"status\":\"ok\"}" : "{\"status\":\"error\"}", "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        if (!srv.engine.alive()) return json_error(res, 503, "the engine is not running", "server_error");
        common_json m = common_json::object();
        m["id"] = kModelName;
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
