# UnbiasedStrata

[Strata](https://github.com/Niko1221/Strata)'s inference engine and llama.cpp's tokenizer, chat-template and
HTTP code, built into one binary that serves Qwen3.8-Flash-Next over an OpenAI-style API. No network is
needed to build or run it: rsync the source, run `make build`, start the binary. No Python, no installer.

## Requirements

| | Build | Run |
| --- | --- | --- |
| Network access | no | no |
| CMake 3.24+, C++20 compiler (tested: g++ 13.3) | yes | no |
| CUDA | toolkit 12.8+ (tested: 12.8; 13.0+ advised for RTX 50) | `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` and the NVIDIA driver |

Hardware, one configuration only: Linux, x86-64 with AVX2, one NVIDIA card with 24 GB or more (RTX 30, 40
or 50), 128 GB RAM. The model can be on a local disk or on a network volume; everything is read into RAM
and VRAM at start, so a slow store costs start-up time only.

## Build

```
rsync -a UnbiasedStrata/ host:UnbiasedStrata/   # or: git clone, then git submodule update --init
ssh host 'cd UnbiasedStrata && make build'      # produces build/unbiased-strata
```

The tree is 6 MB plus the pinned llama.cpp submodule in `third_party/` (about 170 MB checked out). The
build fetches nothing. `make build JOBS=N` sets the job count; `CUDA_ARCHS` defaults to `86;89;120`.

Pascal is an experimental, community build: `make build PASCAL=1` compiles sm_61 (GTX 10-series, Tesla P40;
`CUDA_ARCHS=60` for P100) and turns on the engine's `STRATA_EXPERIMENTAL_SM60` path. It needs a CUDA 12.x
toolkit (CUDA 13 dropped sm_60/sm_70) and is outside the validated configuration above: no tensor cores, so
expect the portable FP32 attention path and much lower throughput, and a card under 24 GB leans on the expert
cache.

## Run

```
./unbiased-strata --model /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
                  --pack /models/pack-iq3-xxs --mtp /models/mtp
```

| | |
| --- | --- |
| Endpoints | `POST /v1/chat/completions` (streaming or not), `GET /v1/models`, `GET /health`, on `127.0.0.1:8080` |
| Options | `--ctx N` (32768), `--threads N` (default: physical cores, or half the CPU quota in a container), `--host`, `--port`, `--mmproj PATH`, `--vision-tokens N` (1024), `--fit-max-tokens` / `--no-fit-max-tokens` (default: on); engine flags after `--` |
| Context | a request whose `max_tokens` would run past `--ctx` has it shortened to the room left; with `--no-fit-max-tokens` it is refused (400) instead |
| Sampling | greedy unless `temperature` > 0 |
| Reasoning | returned as `reasoning_content`; `"reasoning_effort": "none"` turns it off |
| Tools | OpenAI format (`tools`, `tool_calls`) |
| Images | with `--mmproj`, OpenAI `image_url` parts holding a base64 `data:image/...` URL (PNG, JPEG, BMP or GIF), several per message |
| Not supported | more than one GPU, batch requests |

Images: `--mmproj /models/mmproj-Qwen3.8-Flash-Next-BF16.gguf` loads the model's own vision encoder (from the same
Hugging Face repository as the GGUF; 0.9 GB). The encoder runs on the CPU, so the card keeps its expert cache:
about 25 s and 1,000 prompt tokens for a 1-megapixel screenshot on 8 threads, 0.2 s for a small image; a repeated
image is served from a cache. `GET /v1/models` then lists `image` under `input_modalities`. Without `--mmproj` an
image request is refused with a 400.

The engine runs as a child process. When it exits or its stall watchdog fires, the front end exits with a failure
status, for a supervisor such as systemd (`Restart=on-failure`) to restart; `/health` and `/v1/models` answer 503
while the engine is down. Reasoning streams as it is generated; content and tool calls are sent once the whole
message is parsed, so a partial tool-call JSON never changes arguments already sent. `make test-frontend` checks
the child handling and the protocol without a model; `make test-chat-parser MODEL=<shard 1>` checks the chat
template and tool-call parsing against the real GGUF.

## The one supported model

Qwen3.8-Flash-Next (125B parameters, mixture of experts), the full model at IQ3_XXS quantization, from
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
directory `IQ3_XXS`, revision `ed59f920`. No other model, quantization or pruned variant is supported. The
draft (MTP) head is required.

Everything below must be on disk before start. Nothing is fetched.

| What | Size | SHA-256 or source |
| --- | --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (weights) | 47.04 GB | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` (n-gram table), beside shard 1 | 28.80 GB | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| Pack: `index.txt`, `dense.bin`, `native_experts.txt`, `conversions.json`, `expert-profile.bin` | 1.5 GB | made once with upstream's tools; not published yet |
| Draft head: `experts.bin`, `dense.bin`, `dense.txt`, `draft_vocab.bin` | 0.8 GB | made once with upstream's tools; not published yet |

About 78 GB in total. The commands that make the pack and draft head, and the SHA-256 of every file they
produce, are in [UPSTREAM.md](UPSTREAM.md).

## Status

Checked on one machine: RTX 3090, 32 vCPUs of an EPYC 7763, 125 GB RAM, CUDA 12.8, in a container.

| | |
| --- | --- |
| Build | from an rsynced tree with no git metadata; the binary holds no download code |
| Serving | chat, streaming, reasoning, tool calls; a coding agent (pi) wrote and ran a program through it |
| RTX 4090 (24 GB), 82 GB RAM, Debian, bare metal | serving with tool calls and images, through pi (file attachments and clipboard paste) and curl, over several days of use |
| Start from local disk | 19 s with the files in the OS file cache |
| Start from a network volume | 12 min for 76 GB (S3-backed RunPod volume, 120 to 330 MB/s). The volume refused reads twice with "cannot allocate memory"; the loader retried and went on |
| Not tested | RTX 50 cards; a cold start from local disk |

Speed, one run of each, same machine and model files, greedy sampling, reasoning off. Generation is the mean
of 8 different prompts of 400 tokens; prompt speed is the mean of 2 different prompts of about 20,000 tokens.

| | Generation, tokens/s | Prompt, tokens/s |
| --- | --- | --- |
| This fork | 107 (87 to 116) | 2,483 |
| Upstream, with `--pool-workers 15 --ple-io ram` | 78 (61 to 93) | 2,439 |
| Upstream, arguments as its installer writes them | 20 (16 to 21) | 2,368 |

Upstream as installed started 127 CPU workers on the container's 32 vCPUs and read the n-gram table from
disk per token; the middle row removes both. The remaining gap to the fork is not explained yet: the two
builds differ in compile flags and in how the n-gram table is held in RAM, and each row is a single run.

## Known problems

- Text in a message that looks like a control token (`<think>`, `<tool_call>`) is tokenized as that control
  token. Upstream's tokenizer keeps it as plain text.
- Replies are not reproducible token for token between runs or between this fork and upstream, even with
  greedy sampling, so "same output" cannot be checked by comparing text.

## Compared with upstream Strata (6f32ec0)

| | Upstream | This fork |
| --- | --- | --- |
| Engine source (`src/`, `include/`) | 272 files | two files changed, one added (retrying reads from slow stores); cutting it down is the next work |
| Build | clones llama.cpp during CMake | pinned submodule, fetches nothing; one patch, applied by `make build`: tool-call arguments in any order (see UPSTREAM.md) |
| Setup | Python installer downloads the model and builds the pack and draft head | none; files must already exist |
| Server | Python | C++, in the same binary (`frontend/`), with the image encoder from llama.cpp's `mtmd` |
| Python | 33,745 lines | 0 |
| C, C++ and CUDA | 208,254 lines (with tests and the Intel port) | 84,086 lines |
| Tracked files | 851, 27 MB | 292, 6 MB |
| Configurations | several models, NVIDIA, AMD and Intel, Linux and Windows, multi-GPU | one |

## Credits and licence

MIT, as upstream. The engine is the work of Niko1221 and the Strata contributors; see [LICENSE](LICENSE) and
[UPSTREAM.md](UPSTREAM.md). llama.cpp and ggml are MIT. The model files have their own licences.
