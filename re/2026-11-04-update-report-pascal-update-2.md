# Update-Report — UnbiasedStrata → upstream 22abfdb, Pascal-Build sm_61

Card: t_4c5dee40 · Worker: engineer · Date: 2026-11-04 (WET) · Branch: **pascal-update-2** (NOT main)
Repo: `/mnt/nvme/UnbiasedStrata` · Status: **WIP — Build-Gate NOT yet green** (see "Build" + "offene Fragen")

## 1. Chosen Strategy

**Rebase/Reihene-Cherry-Picks onto `upstream/main 22abfdb`** — not a merge. Reason: upstream rebuilt
Sampler/Verify (`src/kernels/cuda/sampler.cu` 4 conflicts, `src/core/mtp.cpp` 2, `include/strata/**`),
our Feature-Hunks are small and mostly dispensable, so a merge would have to re-decide the same 22 conflicts
in reverse direction. The aborted merge on branch `pascal-update` was left untouched and is NOT used.

Result of the series (measured):

```
$ git rev-parse --abbrev-ref HEAD
pascal-update-2
$ git log --oneline -3
4c64d1e Build: PASCAL=1 for the experimental sm_61/sm_60 (GTX 10-series, P40) build
6ae71dc Container: build the unittest image in a Dockerfile; add a P40 compose example
22abfdb DeepSeek-V4.1 docs: prompt-depth table and the IQ1_M set on the A6000 VM
$ git diff --stat upstream/main..HEAD
 .dockerignore            | 16 +++++++++++++
 Dockerfile               | 61 +++++++++++++++++++++++++++++++++++++++++++++++
 Makefile                 |  6 +++++
 README.md                |  6 +++++
 examples/compose/p40.yml | 62 ++++++++++++++++++++++++++++++++++++++++++++++++
 run.sh                   |  1 +
 6 files changed, 152 insertions(+)
```

So: **we sit exactly on upstream 22abfdb plus 2 local commits** (Dockerfile/.dockerignore/P40-compose
example, and the `PASCAL=1` Makefile shorthand). Everything else upstream-wins.

## 2. Conflict decisions (per file)

| File | Decision | Evidence |
|---|---|---|
| `Makefile` | upstream wins (`CUDA_ARCHS ?= 61;75;86;89`, auto `-DSTRATA_EXPERIMENTAL_SM60=ON` for arch<75); our `PASCAL=1` shorthand kept as 6-line alias | `git diff --stat` shows `Makefile +6` vs upstream only |
| `README.md` | upstream wins; 6 added lines = our P40/PASCAL note | `README.md +6` |
| `frontend/engine.hpp` (2 conflicts) | **upstream wins**, our progress-output hunks dropped | cherry-pick of `430af49` was resolved by `git cherry-pick --skip`; no marker left |
| `frontend/main.cpp` (4 conflicts) | **upstream wins** (incl. upstream's `--picture-cache`, `--mmproj*` usage text); our flag-block hunks dropped | `frontend/main.cpp` now byte-identical to upstream (no diff vs upstream) |
| `include/strata/core/verify.hpp`, `include/strata/kernels/sampler.hpp`, `src/core/mtp.cpp`, `src/core/verify.cpp`, `src/kernels/cuda/sampler.cu`, `src/program/generate.cpp` | upstream wins (Konrad's fallback rule: Qwen3.8 kernels/core → upstream) | zero diff vs upstream in these paths |
| `frontend/chat_parser_test.cpp` | upstream deleted it — kept deleted | not present in `git diff --stat` |
| `Dockerfile`, `.dockerignore`, `run.sh`, `examples/compose/p40.yml` | our container work kept (follow-up card t_1e8c918f needs it) | `+61/+16/+1/+62` |

Tree is marker-free (gate):

```
$ grep -rl '^<<<<<<<' --include='*.cpp' --include='*.hpp' --include='*.cu' . | grep -v opencode | wc -l
0
```

## 3. Deliberately dropped (bewusst ausgelassen)

Measured: in the **superproject's own code** (`frontend/`, `src/`, `include/strata/`) there is **no**
trace of our local flags; the remaining hits are llama.cpp's own knobs inside the vendored submodule
and upstream's docs — not our features:

```
reasoning_budget_tokens → only third_party/llama.cpp/{tools/server/*,common/*,tests/*} + docs/DETAILS.md
dry_multiplier          → only third_party/llama.cpp/...
fit-max-tokens / fit_max_tokens → 1 hit each (docs), no frontend flag
sampling-method         → 0
STRATA_PROGRESS         → 0
```

Dropped, with the commit that carried them:

| Feature | Local commit | Why dropped |
|---|---|---|
| `--reasoning-budget-tokens` (hard budget) | `7505a9d` | touches upstream's rebuilt sampler chain; not re-graftable in minutes |
| `--fit-max-tokens` | `d9dc03a`, `13b8eeb` | same sampler/context chain |
| DRY-sampler front-end | `c8c3ca4` | upstream sampler rebuild wins (Konrad rule) |
| verify-windows | `4ad0206` | upstream verify rebuild wins |
| progress output (`STRATA_PROGRESS`) | `430af49` | cherry-pick skipped; upstream's `engine.hpp` kept verbatim (no usage line advertises it) |
| `PASCAL=1` | `6afd3ff` | **kept** as 6-line Makefile shorthand (redundant vs upstream but harmless) |
| Dockerfile + P40 compose | `570b5eb`, `89f653e` | **kept** (commits `6ae71dc`) |

Consequence for the compose card t_1e8c918f: none of the four dropped CLI/env knobs exist in this tree —
`p40-0.yml` must not reference them.

## 4. Submodule (llama.cpp fork pin) — DONE

```
$ git submodule status
 44072a19fb94d0bed63d381d42f6237373f91d54 third_party/llama.cpp (44072a19f)
$ (cd third_party/llama.cpp && git remote -v)
origin https://github.com/robertkarl/llama.cpp.git (fetch)
origin https://github.com/robertkarl/llama.cpp.git (push)
$ git log -1 --format='%H %s'
44072a19f common: DeepSeek V4.1 (deepseek41) chat template detection and DSML tool-call parsing
```

Leading space = checkout matches the superproject pin. The dispatcher's dead-end
(`sync` only rewrites the URL, checkout stuck at ggml-org `3cf03257`) was resolved by
`git fetch --depth 1 origin 44072a19… && git checkout 44072a19…`.

## 5. Tool-call healing / arg-order patch — already upstream, NOT re-applied

`/home/lotherk/patches/local-qwen3-coder-argorder.patch` (Qwen3.8 Flash-Next writes optional args first,
call silently dropped) is **already contained in the fork pin**:

```
$ grep -n 'any order' third_party/llama.cpp/common/parsers/qwen3-coder.cpp
142:                // Accept the arguments in any order, required and optional interleaved. The upstream rule
```

Same comment/prose as our patch → the fork's `44072a1` carries the interleaved
`permute`-rule. Nothing to re-apply; re-applying would duplicate. **Highest-priority item of the card is
therefore satisfied by the pin itself** (observed, not inferred).

## 6. Build — sm_61 (Pascal) — NOT green yet

Attempt 1 (`make build CUDA_ARCHS=61`, log `re/pascal-build-2026-11-04.log`): **exit 2, configure failed**.

```
CMake Error at CMakeLists.txt:1460 (message):
  ggml was not found in : run `git submodule update --init` first
-- Configuring incomplete, errors occurred!
You have changed variables that require your cache to be deleted.
```

Root cause (observed, not guessed): `CMakeLists.txt:1418` sets
`STRATA_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"` as a **CACHE** var, and the error
printed an **empty** dir → the stale `build/CMakeCache.txt` (from our old local CMakeLists) carried
`STRATA_GGML_DIR=""`. The file the check wants exists on disk:

```
$ ls third_party/llama.cpp/ggml/CMakeLists.txt
third_party/llama.cpp/ggml/CMakeLists.txt      # exists → submodule content is fine
```

Fix applied: `rm -rf build` (generated output only) and re-run `make build CUDA_ARCHS=61` with a fresh
cache; log `re/pascal-build-fresh-2026-11-04.log`. At the time of this report the run is **still running /
outcome unknown** — see "offene Fragen".

## 7. Gate status

| Gate | State |
|---|---|
| no merge marker in tree | **met** (0 hits) |
| submodule on `robertkarl/llama.cpp` @ `44072a1` | **met** |
| tool-call healing present | **met** (fork pin) |
| commits on `pascal-update-2`, not `main` | **met** |
| `build/bin/unbiased-strata --help` runs, `--engine` present | **not met** — build did not produce binaries yet |
| `build/bin/strata` + `build/bin/strata-dsv41` built+linked | **not met** |
| `cuobjdump --list-elf build/bin/strata-dsv41 \| grep sm_61` | **not met** |
| Report with strategy / conflicts / dropped / build duration / errors | this file; build duration still open |

Untouched as required: `/mnt/nvme/stack/models/**`, running container `stack-p40-0-1`, image
`unbiased-strata:lotherk-2026-10-15`, `p40-0.yml` (only the compose example `examples/compose/p40.yml`
was added), no gateway restart, nothing downloaded.

## 8. offene Fragen / open lücken

1. **Build outcome unknown at hand-off.** The fresh-cache run (`make build CUDA_ARCHS=61`,
   log `re/pascal-build-fresh-2026-11-04.log`) was still running when this file was written; its exit
   code, any sm_61 compiler errors and the duration are NOT recorded here. Nobody may claim the
   Build-Gate is green until that log ends. Next worker: read the last lines of that log
   (`tail -40 re/pascal-build-fresh-2026-11-04.log`) and check `ls build/bin`.
2. **DSV4.1 on sm_61 unverified.** The `__CUDA_ARCH__ >= 800` branches in the DSV4.1 kernels were not
   inspected or compiled here; whether `strata-dsv41` builds at all for arch 61 (and whether it is runnable
   on the P40 without the Ampere path) is still open — per card rule 4 that is an acceptable outcome, but
   it must be *measured*, not assumed.
3. **`STRATA_EXPERIMENTAL_SM60` propagation.** The Makefile adds the flag for arch<75; whether
   `SM60_FLAG` actually reaches the DSV41 target in the fresh build is open (check the configure line in
   the fresh log).
4. **Uncommitted working-tree state at hand-off:** the fresh build was started after the last commit, so
   `build/` and the two build logs are untracked/ignored; if the reviewer wants the logs committed, they
   live under `re/`.
5. **Dropped-feature re-graft is a decision, not a fact.** Whether `--reasoning-budget-tokens` /
   `--fit-max-tokens` should come back at all (upstream may expose equivalent knobs in the server layer only)
   was not investigated — deliberately out of scope; the compose card decides.
6. `.git/REBASE_HEAD` remains as a stale marker file after the series finished (harmless, no operation
   active, `git status` clean) — removed by nobody.

## 9. Reproduce

```bash
cd /mnt/nvme/UnbiasedStrata
git rev-parse --abbrev-ref HEAD                 # pascal-update-2
git diff --stat upstream/main..HEAD             # 6 files, +152
grep -rl '^<<<<<<<' --include='*.cpp' --include='*.hpp' --include='*.cu' . | wc -l   # 0
git submodule status                            # ' 44072a1…'
rm -rf build && make build CUDA_ARCHS=61        # fresh cache (stale cache = the ggml-not-found trap)
cuobjdump --list-elf build/bin/strata-dsv41 | grep sm_61
build/bin/unbiased-strata --help | grep -- --engine
```
