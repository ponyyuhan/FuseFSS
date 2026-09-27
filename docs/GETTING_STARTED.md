# Getting started

[Project overview](../README.md) · [Reproduction guide](REPRODUCIBILITY.md)

## Choose a starting point

| Goal | Requirements | Output |
|---|---|---|
| Explore operator compilation | CMake 3.24+, C++17 compiler | Two CPU reference/frontend tests |
| Run the GPU core tests | Linux, NVIDIA GPU, compatible CUDA toolkit | Six CTest tests |
| Compare Sigma and FuseFSS | Linux x86-64, C++17/OpenMP compiler, Python 3, CUDA, preferably two GPUs | Paired logs, JSON, and CSV |

Clone the repository:

```bash
git clone https://github.com/ponyyuhan/FuseFSS.git
cd FuseFSS
```

## CPU reference and compiler tests

```bash
cmake -S . -B build_cpu \
  -DFUSEFSS_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build_cpu -j 4
ctest --test-dir build_cpu --output-on-failure
```

Expected test names:

```text
test_fusefss_ref_eval
test_operator_frontend
```

The frontend test uses assertions, so use Debug mode to exercise them. These tests check the reference semantics, operator specification, masking transformations, and post-processing interfaces. They do not launch the two-party GPU protocol.

Useful entry points:

- [`OperatorSpecification`](../include/suf/operator_spec.hpp): interval boundaries, polynomial pieces, predicates, and post-processing.
- [Frontend test examples](../src/tests/test_operator_frontend.cpp): concrete operator specifications and semantic checks.
- [Implementation map](IMPLEMENTATION_MAP.md): how these abstractions connect to the GPU and Sigma paths.

## GPU build

The paper uses two RTX PRO 6000 Blackwell Workstation Edition GPUs, two EPYC 9654 CPUs, and CUDA 13.0. The paired build script uses x86 compiler flags and OpenMP, so use Linux x86-64 for this path.

Check that `nvcc`, `cmake`, and `python3` are on your path and that `nvidia-smi` sees the intended devices. Set `GPU_ARCH` to your GPU's CUDA architecture; `120` is the paper's Blackwell configuration.

```bash
GPU_ARCH=120 bash scripts/build_sigma_pair.sh --variant all
ctest --test-dir build --output-on-failure
```

The script builds the FuseFSS core when needed, then produces both executables:

```text
build/gpu_mpc_upstream/sigma   # Sigma baseline
build/gpu_mpc_vendor/sigma     # Sigma with FuseFSS
```

It also copies each binary into its corresponding Sigma experiment directory. For an explicit core-only build:

```bash
cmake -S . -B build \
  -DFUSEFSS_ENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j 4
```

Use the same architecture in both builds. Run the CPU Debug tests separately when checking assertion-based frontend semantics.

## First paired run

```bash
python3 scripts/repro_eval_latest.py \
  --tasks bert-tiny:128 \
  --runs 1 --warmup 0 \
  --gpu0 0 --gpu1 1 \
  --results-dir results/smoke
```

The runner starts party 0 and party 1 for each variant. It writes:

| File | Contents |
|---|---|
| `results/smoke/summary.csv` | Per-model comparison |
| `results/smoke/results.json` | Measurements and run metadata, including raw byte counts |
| `results/smoke/raw/` | Individual party logs |

For a one-GPU sanity check, append `--single-gpu`; both parties then run on device 0. Resource contention makes this different from the paper's setup.

For paper-style measurements, use `--runs 5 --warmup 1`: five total repetitions, discard the first, report the median of four. The [reproduction guide](REPRODUCIBILITY.md) lists the main models and LLaMA extension.

## Troubleshooting

| Symptom | Check |
|---|---|
| CMake cannot find CUDA | Use `-DFUSEFSS_ENABLE_CUDA=OFF` for CPU tests, or install a CUDA toolkit compatible with your GPU and compiler. |
| `nvcc` rejects `sm_120` | Check the installed toolkit and select the architecture for your device. The paper used CUDA 13.0. |
| Missing `build/gpu_mpc_*/sigma` | Run the paired build script; the core CMake build alone does not create the two Sigma executables. |
| Party 1 cannot use its GPU | Check the visible device IDs, pass `--gpu0`/`--gpu1`, or use `--single-gpu` for a sanity check. |
| A run times out or fails | Read both party logs in the selected results directory. Reduce to `bert-tiny:128` before attempting larger models. |
| A LLaMA run needs substantial memory | Table 8 reports total offline key material, not peak GPU memory; the LLaMA path uses host buffers and streaming. Account for host memory as well as GPU memory. |

Inspect the available options with:

```bash
bash scripts/build_sigma_pair.sh --help
python3 scripts/repro_eval_latest.py --help
```
