# FuseFSS

Artifact repository for:

**FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing**

FuseFSS is an academic proof-of-concept for two-party secure LLM inference in
the input-private, public-model setting. The artifact implements the paper's
scalar nonlinear/helper-operator path and integrates it with Sigma GPU-MPC for
end-to-end transformer inference experiments.

This repository is meant to support the paper's implementation and evaluation
claims. It is not a production private-model serving system.
Due to company policy, a complete public code release must pass a strict
approval process; this artifact contains the implementation needed to support
the paper's claims.

## Repository Layout

```text
include/                 Public and internal FuseFSS headers.
src/                     Compiler, GPU runtime, Sigma bridge, tests.
third_party/EzPC_vendor/ Sigma GPU-MPC with FuseFSS integration.
ezpc_upstream/           Sigma baseline used for comparisons.
scripts/                 Build and evaluation scripts.
docs/                    Artifact notes and claim mapping.
```

## Requirements

- Linux with CUDA-capable NVIDIA GPUs.
- CMake, Python 3, and a CUDA toolkit supported by the target GPU.
- Two GPUs are recommended for the paper-style two-party runs.

The validation machines used `GPU_ARCH=120` for RTX PRO 6000 Blackwell GPUs.
Use the architecture value appropriate for your GPU if it differs.

## Build

Build the FuseFSS core and tests:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Build the Sigma baseline and the Sigma+FuseFSS binary:

```bash
GPU_ARCH=120 bash scripts/build_sigma_pair.sh --variant all
```

## Run A Smoke Test

This runs a small two-party comparison and writes raw logs plus parsed
`results.json` and `summary.csv` files.

```bash
python3 scripts/repro_eval_latest.py \
  --tasks bert-tiny:128 \
  --runs 1 \
  --warmup 0 \
  --results-dir results/smoke
```

## Results

Use the reproduction script to run Sigma and FuseFSS on the same model matrix.
The script records latency, communication, key generation time, key size, GPU
metadata, and the git commit.

```bash
python3 scripts/repro_eval_latest.py \
  --runs 3 \
  --warmup 1 \
  --results-dir results/paper_eval
```

The main output files are:

```text
results/paper_eval/summary.csv
results/paper_eval/results.json
results/paper_eval/raw/
```

## Notes

- The artifact follows the paper's two-server/two-party, semi-honest,
  public-model setting.
- Reported `GB` values follow the Sigma artifact convention
  `bytes / 1024^3`; raw byte counts are kept in `results.json`.
- Llama aliases such as `llama8b` and `llama3.1-8b` map to the Sigma
  `llama3-8b` model id.
- Additional implementation and security notes are in
  [docs/IMPLEMENTATION_MAP.md](docs/IMPLEMENTATION_MAP.md),
  [docs/SECURITY_MODEL.md](docs/SECURITY_MODEL.md), and
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## License

FuseFSS project code is released under the Apache License 2.0. Third-party
components under `third_party/` and `ezpc_upstream/` retain their original
licenses.
