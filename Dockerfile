# syntax=docker/dockerfile:1
#
# UnbiasedStrata on a Pascal card (Tesla P40 / GTX 10-series, sm_61).
#
# The engine and the C++ front end are compiled during docker build into one
# binary, unbiased-strata, which serves the OpenAI-style API.  Nothing is
# downloaded at build or run time: the model shards, the pack and the draft
# (MTP) head must be mounted.  See README.md.
#
# Build:
#   docker build -t unbiased-strata:p40 .
#
# Run (host needs an NVIDIA driver and nvidia-container-toolkit):
#   docker run --rm --gpus '"device=2"' -p 8082:8080 \
#     -v /path/models:/models:ro -v /path/Strata-data:/strata:ro \
#     unbiased-strata:p40 --model /models/...-00001-of-00002.gguf \
#       --pack /strata/pack-iq3-xxs --mtp /strata/mtp \
#       --ctx 32768 --host 0.0.0.0 --port 8080
#
# PASCAL=1 compiles sm_61 and turns on the engine's STRATA_EXPERIMENTAL_SM60
# path.  It needs a CUDA 12.x toolkit (CUDA 13 dropped sm_60/sm_70); 12.8.1 is
# the last 12.x release and still supports Pascal.  Override CUDA_ARCHS to build
# for another card, e.g. --build-arg CUDA_ARCHS=86 for an RTX 30.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

ARG CUDA_ARCHS=61

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# third_party/llama.cpp is the pinned submodule; the Makefile applies
# patches/llama.cpp-qwen3-coder-any-argument-order.patch if it is not already.
# Its .git pointer names a path outside the context, so drop it: git apply then
# runs in no-repository mode, which is all the patch step needs.
RUN rm -rf third_party/llama.cpp/.git \
    && make build PASCAL=1 CUDA_ARCHS="${CUDA_ARCHS}"


FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# ca-certificates/curl are only for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/unbiased-strata /usr/local/bin/unbiased-strata

EXPOSE 8080

# /health answers 503 while the engine is still loading; the long start period
# keeps a slow first load (the model is read into RAM and VRAM) from being
# marked unhealthy and restarted mid-load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
    CMD curl -fs http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/unbiased-strata"]
