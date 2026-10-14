// Exercise request preparation and final tool parsing with the actual GGUF.
// No inference engine or CUDA context is created.
#define main unused_serving_main
#include "main.cpp"
#undef main
#include <cassert>
#include <sstream>

int strata_engine_main(int, char**) { return 99; }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    Server server;
    server.opt.ctx = 196608;
    auto params = llama_model_default_params(); params.vocab_only = true;
    server.model = llama_model_load_from_file(argv[1], params);
    assert(server.model);
    server.vocab = llama_model_get_vocab(server.model);
    server.tmpls = common_chat_templates_init(server.model, "");
    assert(common_chat_templates_get_caps(server.tmpls.get()).at("supports_parallel_tool_calls"));
    auto request = common_json::parse_no_throw(R"({
        "messages":[{"role":"user","content":"Check the files and the date."}],
        "tools":[{"type":"function","function":{"name":"bash","parameters":{
            "type":"object","properties":{"command":{"type":"string"}},"required":["command"]
        }}}]
    })");
    const std::string output = "Reasoning.\n</think>\n\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=bash>\n<parameter=command>\ndate\n</parameter>\n</function>\n</tool_call>";
    Prepared default_request, explicit_request;
    std::string error;
    assert(server.prepare(request, default_request, error));
    auto message = common_chat_parse(output, false, default_request.parser);
    assert(message.tool_calls.size() == 2);
    assert(message.tool_calls[0].arguments == "{\"command\":\"ls\"}");
    assert(message.tool_calls[1].arguments == "{\"command\":\"date\"}");
    request["parallel_tool_calls"] = true;
    assert(server.prepare(request, explicit_request, error));
    assert(explicit_request.ids == default_request.ids);
    assert(common_chat_parse(output, false, explicit_request.parser).tool_calls.size() == 2);
    request["parallel_tool_calls"] = false;
    assert(server.prepare(request, explicit_request, error));
    assert(explicit_request.ids == default_request.ids);
    assert(common_chat_parse(output, false, explicit_request.parser).tool_calls.size() == 1);
    // Arguments in any order: the model writes optional parameters before required ones, and upstream's
    // required-first rule dropped such calls without an error (patches/llama.cpp-qwen3-coder-any-argument-order.patch).
    auto read_request = common_json::parse_no_throw(R"({
        "messages":[{"role":"user","content":"Read the file."}],
        "tools":[{"type":"function","function":{"name":"read","parameters":{
            "type":"object","required":["path"],"properties":{"path":{"type":"string"},
            "offset":{"type":"number"},"limit":{"type":"number"}}
        }}}]
    })");
    Prepared read_prepared;
    assert(server.prepare(read_request, read_prepared, error));
    for (const char* order : {"limit offset path", "offset path limit", "path limit offset", "path"}) {
        std::string text = "Reasoning.\n</think>\n\n<tool_call>\n<function=read>\n";
        std::istringstream names(order);
        for (std::string name; names >> name;)
            text += "<parameter=" + name + ">\n" + (name == "path" ? "/tmp/a.txt" : "40") + "\n</parameter>\n";
        text += "</function>\n</tool_call>";
        auto parsed = common_chat_parse(text, false, read_prepared.parser);
        assert(parsed.tool_calls.size() == 1);
        assert(parsed.tool_calls[0].name == "read");
        assert(parsed.tool_calls[0].arguments.find("\"path\":\"/tmp/a.txt\"") != std::string::npos);
    }
    // Streaming (REGINA-110): the response's text grows a UTF-8 character at a time, as it does token by token
    // in run(); the answer must be published as it grows, not in one piece at the end, and the pieces must add
    // up to the parse of the whole message.  Tool calls are published once, complete.
    struct StreamCase { const char* name; bool thinking; bool tools; std::string output; size_t calls; };
    const std::string call_ls = "<tool_call>\n<function=bash>\n<parameter=command>\nls -l \"a b\" | head -n 3\n"
                                "</parameter>\n</function>\n</tool_call>";
    const std::string answer = "A lighthouse stands on the cape: caf\xC3\xA9 lights, 2 < 3 and <b>bold</b>.\n\n"
                               "```json\n{\"a\": [1, 2], \"s\": \"x\\ny\"}\n```\nIt writes <tool and <function= as text.";
    const StreamCase cases[] = {
        {"reasoning then answer", true, false, "The user asks.\nTwo steps.\n</think>\n\n" + answer, 0},
        {"answer, thinking off", false, false, answer, 0},
        {"answer, tools declared", true, true, "No tool is needed.\n</think>\n\n" + answer, 0},
        {"answer, tools declared, thinking off", false, true, answer, 0},
        {"text then a tool call", true, true, "List first.\n</think>\n\nI will list the files.\n\n" + call_ls, 1},
        {"a tool call alone", false, true, call_ls, 1},
        {"two tool calls", true, true, "Both.\n</think>\n\n" + call_ls + "\n" + call_ls, 2},
    };
    for (const StreamCase& c : cases) {
        auto body = common_json::parse_no_throw(R"({"messages":[{"role":"user","content":"Describe a lighthouse."}],
            "stream":true})");
        if (c.tools) body["tools"] = request.at("tools");
        body["chat_template_kwargs"] = common_json::object();
        body["chat_template_kwargs"]["enable_thinking"] = c.thinking;
        Prepared prepared;
        assert(server.prepare(body, prepared, error));
        const common_chat_msg whole = common_chat_parse(c.output, false, prepared.parser);
        StreamParse parse;
        std::string content, reasoning;
        size_t content_deltas = 0, reasoning_deltas = 0, call_deltas = 0, steps = 0;
        bool calls_only_at_end = true;
        auto take = [&](const std::vector<common_chat_msg_diff>& diffs, bool partial) {
            for (const auto& d : diffs) {
                if (!d.content_delta.empty()) { content += d.content_delta; ++content_deltas; }
                if (!d.reasoning_content_delta.empty()) { reasoning += d.reasoning_content_delta; ++reasoning_deltas; }
                if (d.tool_call_index != std::string::npos) { ++call_deltas; if (partial) calls_only_at_end = false; }
            }
        };
        for (size_t n = 1; n < c.output.size(); ++n) {
            if (utf8_complete_prefix(c.output.substr(0, n)) != n) continue;
            take(parse.step(c.output.substr(0, n), true, prepared.parser), true);
            ++steps;
        }
        take(parse.step(c.output, false, prepared.parser), false);
        std::printf("stream case '%s': %zu steps, %zu reasoning deltas, %zu content deltas, %zu tool-call deltas\n",
                    c.name, steps, reasoning_deltas, content_deltas, call_deltas);
        if (std::getenv("STREAM_TEST_VERBOSE"))
            std::printf("  whole content [%s]\n  streamed      [%s]\n", whole.content.c_str(), content.c_str());
        assert(reasoning == whole.reasoning_content);
        // the streamed answer is the whole one; before a tool call it may keep the white space the parser trims
        assert(content.compare(0, whole.content.size(), whole.content) == 0);
        assert(content.find_first_not_of(" \n", whole.content.size()) == std::string::npos);
        assert(whole.content.size() < 20 || content_deltas * 2 >= whole.content.size());
        assert(whole.reasoning_content.size() < 10 || reasoning_deltas * 2 >= whole.reasoning_content.size());
        assert(whole.tool_calls.size() == c.calls && parse.msg.tool_calls.size() == c.calls);
        assert(calls_only_at_end && call_deltas == c.calls);
        for (size_t i = 0; i < c.calls; ++i) {
            assert(parse.msg.tool_calls[i].arguments == "{\"command\":\"ls -l \\\"a b\\\" | head -n 3\"}");
            assert(!parse.msg.tool_calls[i].id.empty());
        }
    }
    // #123: reasoning_budget_tokens / reasoning_budget_message - the request's value wins over the command line's,
    // a bad value is refused, thinking off leaves no budget, and the template's end-of-thinking tags are captured.
    server.opt.reasoning_budget = 64;
    server.opt.reasoning_budget_message = "server default";
    Prepared budgeted;
    assert(server.prepare(request, budgeted, error));
    assert(budgeted.thinking);
    assert(budgeted.reasoning_budget == 64);
    assert(budgeted.reasoning_budget_message == "server default");
    assert(!budgeted.thinking_end_tags.empty());
    request["reasoning_budget_tokens"] = 128;
    request["reasoning_budget_message"] = "You exhausted your reasoning tokens, come to a conclusion, now!";
    assert(server.prepare(request, budgeted, error));
    assert(budgeted.reasoning_budget == 128);
    assert(budgeted.reasoning_budget_message == "You exhausted your reasoning tokens, come to a conclusion, now!");
    request.erase("reasoning_budget_message");
    request.erase("reasoning_budget_tokens");
    request["reasoning_budget_tokens"] = -1;
    assert(!server.prepare(request, budgeted, error));
    request["reasoning_budget_tokens"] = "lots";
    assert(!server.prepare(request, budgeted, error));
    request.erase("reasoning_budget_tokens");
    request["reasoning_effort"] = "none";
    assert(server.prepare(request, budgeted, error));
    assert(!budgeted.thinking && budgeted.reasoning_budget == 0);
    request.erase("reasoning_effort");
    // Sampling header: the launch-time defaults, overridden by a request's own fields; greedy is preserved;
    // penalties apply even in greedy and get llama.cpp's 64-token window when none is set.
    {
        const common_json empty = common_json::parse_no_throw(R"({"messages":[]})");
        Options neutral;
        assert(sampling_keys(empty, neutral, server.vocab).empty());   // greedy, no penalties: exactly the old header

        Options tuned = neutral;
        tuned.temperature = 0.7; tuned.top_p = 0.9; tuned.top_k = 40; tuned.min_p = 0.05;
        tuned.repeat_penalty = 1.05; tuned.presence_penalty = 0.3; tuned.frequency_penalty = 0.2;
        tuned.repeat_last_n = 256; tuned.seed = 42;
        const std::string k = sampling_keys(empty, tuned, server.vocab);
        for (const char* s : {" temperature=0.7", " top_p=0.9", " top_k=40", " seed=42", " min_p=0.05",
                              " penalty_repeat=1.05", " penalty_present=0.3", " penalty_freq=0.2",
                              " penalty_last_n=256"})
            assert(k.find(s) != std::string::npos);

        // a request's own fields win; the ones it omits keep the launch default
        const common_json override_body = common_json::parse_no_throw(
            R"({"messages":[],"temperature":0.2,"top_k":8,"repeat_penalty":1.1,"repeat_last_n":512})");
        const std::string ko = sampling_keys(override_body, tuned, server.vocab);
        for (const char* s : {" temperature=0.2", " top_k=8", " top_p=0.9", " penalty_repeat=1.1",
                              " penalty_last_n=512"})
            assert(ko.find(s) != std::string::npos);

        // the greedy path honors penalties too, so a penalty applies with no temperature
        const common_json penalty_body = common_json::parse_no_throw(R"({"messages":[],"presence_penalty":0.5})");
        const std::string kp = sampling_keys(penalty_body, neutral, server.vocab);
        assert(kp.find("temperature") == std::string::npos);
        assert(kp.find(" penalty_present=0.5") != std::string::npos);
        assert(kp.find(" penalty_last_n=64") != std::string::npos);

        // DRY: emitted only when the multiplier and the window are both set; the default breakers are tokenized
        // here (the engine has no tokenizer) and encoded as ids; a request's own list wins.
        Options dry = neutral;
        dry.dry_multiplier = 0.8; dry.dry_base = 1.75; dry.dry_allowed_length = 2; dry.dry_penalty_last_n = 256;
        const std::string kd = sampling_keys(empty, dry, server.vocab);
        for (const char* s : {" dry_multiplier=0.8", " dry_base=1.75", " dry_allowed_length=2",
                              " dry_penalty_last_n=256", " dry_breakers="})
            assert(kd.find(s) != std::string::npos);
        assert(sampling_keys(empty, neutral, server.vocab).find("dry_") == std::string::npos);
        const common_json dry_body = common_json::parse_no_throw(
            R"({"messages":[],"dry_multiplier":0.5,"dry_penalty_last_n":128,"dry_sequence_breakers":["\n"]})");
        const std::string kb = sampling_keys(dry_body, neutral, server.vocab);
        assert(kb.find(" dry_multiplier=0.5") != std::string::npos);
        assert(kb.find(" dry_penalty_last_n=128") != std::string::npos);
        assert(kb.find(" dry_breakers=") != std::string::npos);
    }
    llama_model_free(server.model);
    std::puts("PASS: omitted parallel_tool_calls uses the model capability; explicit false remains honored; "
              "tool arguments parse in any order; the answer streams as it grows and tool calls arrive whole; "
              "reasoning_budget_tokens and its message override the command line; the sampling header honors the "
              "launch defaults, a request's overrides, greedy and the penalty window");
}
