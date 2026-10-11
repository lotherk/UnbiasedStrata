# UnbiasedStrata build.  Needs CMake 3.24+, a C++20 compiler and the CUDA toolkit.  Nothing is downloaded.
BUILD_DIR  ?= build
# Pascal/Volta are a community, experimental build: `make build PASCAL=1` compiles sm_61 (GTX 10-series,
# Tesla P40; sm_60/P100 works too, override with CUDA_ARCHS=60) and turns on the engine's
# STRATA_EXPERIMENTAL_SM60 path (see README.md and src/core/device.cu).  It needs a CUDA 12.x toolkit:
# CUDA 13 dropped sm_60/sm_70, so the check below fails early with the reason instead of inside nvcc.
PASCAL     ?=
ifeq ($(PASCAL),1)
CUDA_ARCHS ?= 61
PASCAL_CMAKE_FLAGS := -DSTRATA_EXPERIMENTAL_SM60=ON
NVCC       ?= $(shell command -v nvcc 2>/dev/null)
NVCC_MAJOR := $(shell $(NVCC) --version 2>/dev/null | sed -n 's/.*release \([0-9]*\).*/\1/p' | head -n1)
ifneq ($(NVCC_MAJOR),)
ifeq ($(shell [ $(NVCC_MAJOR) -ge 13 ] && echo yes),yes)
$(error PASCAL=1 needs a CUDA 12.x toolkit; nvcc $(NVCC_MAJOR) dropped sm_60/sm_70)
endif
endif
else
# RTX 30 (86), RTX 40 (89) and RTX 50 (120) in one binary.  sm_120 needs CUDA 12.8+, and upstream recommends 13.0+.
CUDA_ARCHS ?= 86;89;120
endif
# capped: inside a container nproc reports the host's CPUs, not the container's quota
JOBS       ?= $(shell n=$$(nproc 2>/dev/null || echo 4); [ $$n -gt 16 ] && n=16; echo $$n)

.PHONY: build clean

build:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || \
	  { echo "third_party/llama.cpp is empty: run 'git submodule update --init' first"; exit 1; }
	@git -C third_party/llama.cpp apply --check --reverse ../../patches/llama.cpp-qwen3-coder-any-argument-order.patch 2>/dev/null || \
	  git -C third_party/llama.cpp apply ../../patches/llama.cpp-qwen3-coder-any-argument-order.patch
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON \
	      -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" \
	      $(PASCAL_CMAKE_FLAGS)
	cmake --build $(BUILD_DIR) -j $(JOBS) --target unbiased-strata
	@echo "built: $(BUILD_DIR)/unbiased-strata"

clean:
	rm -rf $(BUILD_DIR)

# Protocol and child supervision regressions; no CUDA/model required.
.PHONY: test-frontend
test-frontend:
	mkdir -p $(BUILD_DIR)/frontend-tests
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/child_watch_test.cpp -o $(BUILD_DIR)/frontend-tests/child-watch-test
	$(BUILD_DIR)/frontend-tests/child-watch-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/engine_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-test
	$(BUILD_DIR)/frontend-tests/engine-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror -pthread frontend/engine_backpressure_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-backpressure-test
	$(BUILD_DIR)/frontend-tests/engine-backpressure-test

# Model-template compatibility; CPU only, after the regular build produced llama libraries.
.PHONY: test-chat-parser
test-chat-parser:
	@test -n "$(MODEL)" || { echo 'set MODEL to the first GGUF shard'; exit 2; }
	mkdir -p $(BUILD_DIR)/frontend-tests
	$(CXX) -std=c++20 -O0 -Ifrontend -Ithird_party/llama.cpp/include -Ithird_party/llama.cpp/common \
	  -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/vendor -Ithird_party/llama.cpp/tools/mtmd \
	  frontend/chat_parser_test.cpp \
	  -o $(BUILD_DIR)/frontend-tests/chat-parser-test $(BUILD_DIR)/llama.cpp/tools/mtmd/libmtmd.a \
	  $(BUILD_DIR)/llama.cpp/vendor/hash/libvendor-hash.a $(BUILD_DIR)/llama.cpp/common/libllama-common.a \
	  $(BUILD_DIR)/llama.cpp/src/libllama.a $(BUILD_DIR)/llama.cpp/vendor/cpp-httplib/libcpp-httplib.a \
	  $(BUILD_DIR)/ggml/src/libggml.a $(BUILD_DIR)/ggml/src/libggml-cpu.a $(BUILD_DIR)/ggml/src/libggml-base.a \
	  $(BUILD_DIR)/llama.cpp/common/libllama-common-base.a -lpthread -ldl -lm
	$(BUILD_DIR)/frontend-tests/chat-parser-test "$(MODEL)"
