# External Integrations

**Analysis Date:** 2026-07-27

## Summary

**This project has no external integrations.** It is a fully offline, self-contained C command-line pipeline with zero network calls, zero third-party API/SDK usage, zero database connections, and zero cloud/hosting dependencies. Verified by exhaustive search of `src/` and `include/` for socket, HTTP, curl, and database (`sqlite`/`mysql`/`postgres`) symbols — none found. All I/O is local filesystem read/write.

## APIs & External Services

**None.** No REST/GraphQL clients, no SDK imports (no AWS/Stripe/Supabase/etc.), no `curl`/`libcurl` usage anywhere in the codebase.

## Data Storage

**Databases:**
- None. No SQL or NoSQL database of any kind. No ORM, no DB client library.

**File Storage:**
- Local filesystem only. All data is read from and written to the local working directory:
  - Input: WAV audio files under 5 class subdirectories (`saudavel/`, `laringite/`, `disfonia_psicogênica/`, `disfonia_funcional/`, `edema_de_reinke/`) enumerated via POSIX `dirent.h`/`opendir` in `src/dataset.c`.
  - Input metadata: `overview_merged.csv` (patient demographics/diagnosis), parsed by a hand-rolled RFC 4180-compliant CSV parser in `src/csv_parser.c` (no external CSV library).
  - Output/cache: `results/features.csv` (cached ~1098×251 feature matrix, read/written in `src/main.c`'s `features_load_csv`/feature-extraction path), `results/metrics_global.csv`, `results/learning_curves.csv`, `results/roc_curves.csv`, `results/pr_curves.csv`, `results/baselines.csv`, `results/feature_importance.csv`, `results/train_log_v*.txt`.
  - Model artifacts: `models/mlp_fold{0-4}.bin`, `models/norm_fold{0-4}.bin`, `models/selected_fold{0-4}.bin`, `models/best_model.bin`, `models/best_norm.bin`, `models/best_selected.bin` — custom binary serialization (no format library, e.g., no protobuf/msgpack).

**Caching:**
- Simple file-existence-based cache: `results/features.csv` is loaded instead of re-extracting from WAVs if present, with column-count validation against `TOTAL_FEATURES` in `include/config.h` (auto-invalidates on feature-set changes). This is application-level caching, not a caching service (no Redis/Memcached).

## Authentication & Identity

**Auth Provider:**
- None. This is a local CLI tool with no user accounts, sessions, or authentication of any kind.

## Monitoring & Observability

**Error Tracking:**
- None. No Sentry, Rollbar, or similar service. Errors are handled via return codes and `log_error()`/`log_info()` helpers in `src/utils.c`, which print to stdout/stderr.

**Logs:**
- Local file + console logging only. Training/extraction runs write plaintext logs (e.g., `results/train_log_v30.txt`, `results/train_log_v31.txt`, `results/extract_log.txt`) via custom logging functions in `src/utils.c`; no structured logging framework, no log-shipping.

## CI/CD & Deployment

**Hosting:**
- None. Not deployed anywhere; run manually via `make`/`./build/vocal_detect` on a local or lab machine.

**CI Pipeline:**
- None detected. No `.github/workflows/`, `.gitlab-ci.yml`, `Jenkinsfile`, or other CI configuration found in the repository.

## Environment Configuration

**Required env vars:**
- None. No `getenv()` calls in the codebase. All configuration is compile-time (`include/config.h`) or CLI positional arguments (`extract|train|full [base_dir]`).

**Secrets location:**
- Not applicable — no secrets, API keys, or credentials are used anywhere in this project. No `.env` files present in the repository.

## Webhooks & Callbacks

**Incoming:**
- None — no server component exists (this is a batch CLI tool, not a listening service).

**Outgoing:**
- None.

## External Dataset Dependency (non-code)

- The pipeline is designed to operate on the **SVD (Saarbrücken Voice Database)**, an external academic voice-pathology dataset. This is a *data* dependency, not a live integration: WAV files must be manually downloaded/placed in the working directory (see `README.md` for attribution: Barry, W.J. & Pützer, M. (2007), Institute of Phonetics, Saarland University). No API call or automated download fetches this data; it is excluded from version control (`saudavel/`, `laringite/`, `disfonia_psicogênica/`, `disfonia_funcional/`, `edema_de_reinke/` are all listed in `.gitignore`).

---

*Integration audit: 2026-07-27*
