# Contributing to FuseFSS

Bug reports, documentation improvements, and focused contributions are welcome.

## Report an issue

Use [GitHub Issues](https://github.com/ponyyuhan/FuseFSS/issues). Include the commit, command, relevant error, operating system, compiler, and (for GPU issues) CUDA version and GPU model. For benchmark questions, include the model, sequence length, run/warmup counts, and whether both parties shared a GPU. Remove credentials and private inputs from logs before sharing them.

## Develop and test

Start with the [setup guide](docs/GETTING_STARTED.md). Run the CPU tests in Debug mode so frontend assertions stay active:

```bash
cmake -S . -B build_cpu \
  -DFUSEFSS_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build_cpu -j 4
ctest --test-dir build_cpu --output-on-failure
```

For GPU changes, also run the relevant CUDA tests and a small paired comparison from the [reproduction guide](docs/REPRODUCIBILITY.md). Describe which checks you ran and any hardware-dependent checks you could not run.

Keep changes focused and explain the behavior they change. Preserve fresh per-wire masking, mask-independent public shapes, and the private-model guards documented in the [security model](docs/SECURITY_MODEL.md). Distinguish optimized-path performance measurements from generic-path semantic checks.

Avoid committing build outputs, generated keys, datasets, or local experiment logs. Third-party code retains its original license and attribution.
