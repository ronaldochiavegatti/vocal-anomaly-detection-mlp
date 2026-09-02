# Technology Stack

**Analysis Date:** 2026-07-27

## Languages

**Primary:**
- C (C99 standard, `-std=c99`) - 100% of the codebase: `src/*.c` (19 files, ~5,666 total lines across `src/` and `include/`), `include/*.h` (17 headers)

**Secondary:**
- None. No Python, shell scripts, or other languages are part of the build/runtime pipeline. (Non-code artifacts in the repo root — `.docx`/`.pptx` files, `overview_merged.csv` — are academic/documentation/data assets, not part of the toolchain.)

## Runtime

**Environment:**
- Native compiled binary (ELF executable), no VM/interpreter. Built and run directly on Linux (developed/tested on Ubuntu 24.04, gcc 13.3.0, kernel 6.17).
- No containerization (no `Dockerfile`, no `docker-compose.yml` in repo).

**Package Manager:**
- None. There is no language-level package manager (no `pip`, `npm`, `cargo`, `conan`, `vcpkg`). All dependencies are system libraries linked directly by the compiler (`libm`, OpenMP runtime `libgomp`).
- Lockfile: not applicable — no dependency manifest exists.

## Frameworks

**Core:**
- None. No ML/DL framework (no TensorFlow, PyTorch, ONNX, scikit-learn). The Multi-Layer Perceptron is hand-implemented from scratch in `src/mlp.c` / `src/mlp_train.c` (forward pass with LeakyReLU + Dropout + Softmax, manual backprop, Adam optimizer, cosine LR annealing, gradient clipping).
- Baseline classifiers (kNN in `src/knn.c`, logistic regression in `src/logreg.c`) are also hand-implemented, not from a library.

**Testing:**
- None detected. No unit test framework (no CUnit, Check, Unity). The `Makefile` defines a `test` target (`./build/vocal_detect test`) but this mode is explicitly **not implemented** in `src/main.c` (per `CLAUDE.md`). There is no `tests/` directory and no automated test suite.

**Build/Dev:**
- GNU Make (`Makefile`, GNU Make syntax with `$(wildcard ...)`, pattern rules) - drives the entire build.
- GCC 13.3.0 (`gcc`) - sole compiler; flags: `-O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp`.
- OpenMP 4.5 (`_OPENMP 201511`, via `-fopenmp`) - used for parallel feature extraction (`#pragma omp parallel for reduction(+:errors)` in `src/feature_extract.c`), giving ~6.5× speedup (257s → ~39-42s for 1098 patients).

## Key Dependencies

**Critical:**
- `libm` (math library, linked via `-lm`) - all DSP math (FFT, autocorrelation, log, trig, `erfc` for McNemar test p-values).
- `libgomp` (OpenMP runtime, linked via `-fopenmp`) - parallelizes the per-patient feature-extraction loop across CPU cores.
- Standard C library (`libc`) headers used throughout: `stdio.h`, `stdlib.h`, `string.h`, `math.h`, `time.h`, `stdint.h`, `stddef.h`, `stdarg.h`, `dirent.h`, `sys/stat.h`. No third-party C libraries (no libsndfile, no FFTW, no BLAS/LAPACK) — WAV parsing, FFT, and CSV parsing are all custom implementations.

**Infrastructure:**
- None. No message queues, caches, or service dependencies. The only "infrastructure" is the local filesystem (input WAV directories, `overview_merged.csv`, and generated `results/`/`models/` directories).

## Configuration

**Environment:**
- No environment variables are read anywhere in `src/` (no `getenv` calls found). All tunables are compile-time constants.
- All configuration lives in a single header: `include/config.h` — paths, audio parameters, class definitions, feature counts, MLP architecture, hyperparameters, class weights, random seed. Changing any of these requires recompilation (`make clean && make`).
- Command-line arguments select pipeline mode: `./build/vocal_detect {extract|train|full} [base_dir]` (parsed in `src/main.c`); `base_dir` defaults to the current working directory if omitted.

**Build:**
- `Makefile` (root) — single build config file; no CMake, no Meson, no Autotools.
- No `tsconfig.json`/`eslint.config`/`package.json`-equivalent exists; this is a pure C project with no auxiliary tool configs beyond the Makefile and `include/config.h`.

## Platform Requirements

**Development:**
- Linux (developed on Ubuntu 24.04 LTS, kernel 6.17).
- `gcc` supporting C99 and OpenMP (tested with gcc 13.3.0).
- `make` (GNU Make).
- Multi-core CPU recommended (OpenMP feature extraction scales with core count).
- Local copies of the SVD (Saarbrücken Voice Database) WAV files, organized into 5 class directories (`saudavel/`, `laringite/`, `disfonia_psicogênica/`, `disfonia_funcional/`, `edema_de_reinke/`) plus `overview_merged.csv` metadata, placed in the working directory (not checked into git — see `.gitignore`).
- `results/` and `models/` directories must be created manually (`mkdir -p results models`) before running.

**Production:**
- No deployment target — this is a research/academic pipeline run locally via CLI (`./build/vocal_detect train`), not a deployed service. Output artifacts (`results/*.csv`, `models/*.bin`) are consumed manually/offline for analysis, not served.
- Portable to any POSIX-like system with a C99 compiler and OpenMP support (Linux primarily; not verified on macOS/Windows). Uses POSIX-specific APIs (`dirent.h`, `sys/stat.h`, `_POSIX_C_SOURCE 200809L` in `src/dataset.c`), so Windows would require WSL/MinGW/Cygwin.

---

*Stack analysis: 2026-07-27*
