# Public artifact scope

This repository contains the implementation and evaluation artifact for [FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing](https://openreview.net/forum?id=WpUpj8DrVB).

It includes the operator specification and compiler, GPU runtime, Sigma integration, tests, and evaluation scripts. FuseFSS targets scalar nonlinear/helper operators; tensor-level operations continue to use Sigma. The [implementation map](IMPLEMENTATION_MAP.md) explains the optimized and generic execution paths.

The original artifact release note states that, due to company policy, a complete public code release must pass a strict approval process. This artifact contains the implementation needed to support the paper's claims.

The artifact is an academic proof of concept in the input-private, public-model setting, not a production private-model serving system. See the [security model](SECURITY_MODEL.md) for the protocol assumptions and existing private-model guards.

## Licensing

FuseFSS project code uses the [Apache License 2.0](../LICENSE). Code under `third_party/` and `ezpc_upstream/` retains its original licenses. Preserve upstream notices when modifying or redistributing those components.
