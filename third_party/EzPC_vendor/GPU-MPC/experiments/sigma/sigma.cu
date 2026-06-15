// Author: Neha Jawalkar
// Copyright:
//
// Copyright (c) 2024 Microsoft Research
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cfloat>
#ifndef FLT_MIN
#define FLT_MIN __FLT_MIN__
#endif
#ifndef DBL_MAX
#define DBL_MAX __DBL_MAX__
#endif
#ifndef DBL_MIN
#define DBL_MIN __DBL_MIN__
#endif

#include <sytorch/module.h>
#include <sytorch/utils.h>
#include <sytorch/backend/cleartext.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include "gpt2.h"
#include "bert.h"
#include "llama2.h"
#include "backend/sigma.h"
#ifdef SUF_HAVE_CUDA
#include "suf/sigma_bridge.hpp"
#include "suf/operator_spec.hpp"
#include "fss/gpu_and.h"
#include "fss/gpu_lut.h"
#endif

template u64 *gpuKeygenMul<u64>(u8 **key_as_bytes, int party, int bw, int scale, int N,
                                u64 *d_mask_A, u64 *d_mask_B, TruncateType t,
                                AESGlobalContext *gaes);
template u64 *gpuMul<u64>(SigmaPeer *peer, int party, int bw, int scale, int N,
                          GPUMulKey<u64> k, u64 *d_X, u64 *d_Y, TruncateType t,
                          AESGlobalContext *gaes, Stats *s);
template u64 *randomGEOnGpu<u64>(const u64 n, int bw);
#ifdef SUF_HAVE_CUDA
template u64 *gpuKeyGenAnd<u64>(u8 **key_as_bytes, int party, int bout, int N,
                                u64 *d_b0, u64 *d_b1);
template u64 *gpuKeyGenVectorLUT<u16, u64>(uint8_t **key_as_bytes, int party,
                                           int bin, int bout, int outWords, int N,
                                           u16 *d_rin, AESGlobalContext *gaes);
template u64 *gpuDpfVectorLUT<u16, u64>(GPUVectorLUTKey<u64> k0, SigmaPeer *peer,
                                        int party, u16 *d_X, u64 *d_tab,
                                        AESGlobalContext *g, Stats *s,
                                        bool opMasked);
#endif

inline std::string toGB(u64 bytes)
{
    return std::to_string(bytes) + " B (" + std::to_string((float)bytes / (1024.0f * 1024.0f * 1024.0f)) + " GB)";
}

inline const char *fusefssEnvValue(const char *name)
{
    if (std::strncmp(name, "SUF_", 4) == 0)
    {
        std::string alias = "FUSEFSS_";
        alias += name + 4;
        const char *v = std::getenv(alias.c_str());
        if (v && v[0])
            return v;
    }
    return std::getenv(name);
}

inline bool envFlag(const char *name)
{
    const char *v = fusefssEnvValue(name);
    return v && v[0] && std::atoi(v) != 0;
}

inline u64 envU64(const char *name, u64 fallback)
{
    const char *v = fusefssEnvValue(name);
    if (!v || !v[0])
        return fallback;
    return std::strtoull(v, nullptr, 10);
}

static void writeTensorBin(const std::string &path, const Tensor<u64> &t)
{
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(t.data), t.size() * sizeof(u64));
}

static void writeTensorMeta(const std::string &path, const Tensor<u64> &t, const std::string &model, u64 scale, int bw)
{
    std::ofstream meta(path);
    meta << "{\n";
    meta << "  \"model\": \"" << model << "\",\n";
    meta << "  \"shape\": [";
    for (size_t i = 0; i < t.shape.size(); ++i)
    {
        meta << t.shape[i];
        if (i + 1 < t.shape.size())
            meta << ", ";
    }
    meta << "],\n";
    meta << "  \"scale\": " << scale << ",\n";
    meta << "  \"bw\": " << bw << ",\n";
    meta << "  \"dtype\": \"u64\",\n";
    meta << "  \"signed\": true\n";
    meta << "}\n";
}

#ifdef SUF_HAVE_CUDA
static suf::BoolExpr fusefssPredExprForCanary(int pred_index)
{
    suf::BoolExpr e;
    e.nodes.push_back(suf::BoolNode{suf::BoolNode::Kind::PRED, -1, -1, pred_index});
    e.root = 0;
    return e;
}

static suf::OperatorSpecification makeFuseFSSAffineReluPhiSpec()
{
    suf::OperatorSpecification spec;
    spec.in_bits = 8;
    spec.boundaries = {0, 128};

    suf::Predicate msb;
    msb.kind = suf::PredKind::MSB;
    spec.predicates.push_back(msb);

    spec.pieces.resize(2);
    spec.pieces[0].polys = {suf::Polynomial{{1}}, suf::Polynomial{{3}}};
    spec.pieces[0].aux_words = {17};
    spec.pieces[0].bool_outputs = {fusefssPredExprForCanary(0)};
    spec.pieces[1].polys = {suf::Polynomial{{0}}, suf::Polynomial{{5}}};
    spec.pieces[1].aux_words = {29};
    spec.pieces[1].bool_outputs = {fusefssPredExprForCanary(0)};

    suf::PostprocessArithExpr x_out;
    x_out.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::X, -1, -1, -1, 0});
    x_out.root = 0;
    spec.postprocess.arith_exprs.push_back(x_out);

    suf::PostprocessArithExpr a_out;
    a_out.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
    a_out.root = 0;
    spec.postprocess.arith_exprs.push_back(a_out);

    suf::PostprocessArithExpr b_out;
    b_out.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::POLY_OUT, -1, -1, 1, 0});
    b_out.root = 0;
    spec.postprocess.arith_exprs.push_back(b_out);

    suf::PostprocessArithExpr prod;
    prod.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
    prod.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::X, -1, -1, -1, 0});
    prod.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::MUL, 0, 1, -1, 0});
    prod.root = 2;
    spec.postprocess.arith_exprs.push_back(prod);

    suf::PostprocessArithExpr y;
    y.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
    y.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::X, -1, -1, -1, 0});
    y.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::MUL, 0, 1, -1, 0});
    y.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::POLY_OUT, -1, -1, 1, 0});
    y.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::ADD, 2, 3, -1, 0});
    y.root = 4;
    spec.postprocess.arith_exprs.push_back(y);
    spec.postprocess.arithmetic_outputs = {0, 1, 2, 3, 4};

    suf::PostprocessBoolExpr z;
    z.nodes.push_back(suf::PostprocessBoolNode{suf::PostprocessBoolOp::BOOL_OUT, -1, -1, 0, 0});
    z.root = 0;
    spec.postprocess.bool_exprs.push_back(z);
    spec.postprocess.boolean_outputs = {0};
    return spec;
}

static suf::OperatorSpecification makeFuseFSSKappaA2BBitSpec()
{
    suf::OperatorSpecification spec;
    spec.in_bits = 8;
    spec.boundaries = {0};
    spec.pieces.resize(1);
    spec.pieces[0].polys = {suf::Polynomial{{1}}};

    suf::PostprocessArithExpr x_expr;
    x_expr.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::X, -1, -1, -1, 0});
    x_expr.root = 0;
    spec.postprocess.arith_exprs.push_back(x_expr);

    suf::PostprocessArithExpr y_expr;
    y_expr.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::POLY_OUT, -1, -1, 0, 0});
    y_expr.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::KAPPA_A, -1, -1, 0, 0});
    y_expr.nodes.push_back(suf::PostprocessArithNode{suf::PostprocessArithOp::ADD, 0, 1, -1, 0});
    y_expr.root = 2;
    spec.postprocess.arith_exprs.push_back(y_expr);

    suf::PostprocessBoolExpr z_expr;
    z_expr.nodes.push_back(suf::PostprocessBoolNode{suf::PostprocessBoolOp::KAPPA_B, -1, -1, 0, 0});
    suf::PostprocessBoolNode bit3;
    bit3.op = suf::PostprocessBoolOp::A2B;
    bit3.index = 0;
    bit3.bit_index = 3;
    z_expr.nodes.push_back(bit3);
    z_expr.nodes.push_back(suf::PostprocessBoolNode{suf::PostprocessBoolOp::XOR, 0, 1, -1, 0});
    z_expr.root = 2;
    spec.postprocess.bool_exprs.push_back(z_expr);

    spec.postprocess.arithmetic_outputs = {1};
    spec.postprocess.boolean_outputs = {0};
    return spec;
}

static suf::OperatorSpecification makeFuseFSSBoolOrSpec()
{
    suf::OperatorSpecification spec;
    spec.in_bits = 8;
    spec.boundaries = {0, 128};
    suf::Predicate msb;
    msb.kind = suf::PredKind::MSB;
    spec.predicates.push_back(msb);
    spec.pieces.resize(2);
    spec.pieces[0].polys = {suf::Polynomial{{0}}};
    spec.pieces[0].bool_outputs = {fusefssPredExprForCanary(0)};
    spec.pieces[1].polys = {suf::Polynomial{{0}}};
    spec.pieces[1].bool_outputs = {fusefssPredExprForCanary(0)};

    suf::PostprocessBoolExpr z;
    z.nodes.push_back(suf::PostprocessBoolNode{suf::PostprocessBoolOp::BOOL_OUT, -1, -1, 0, 0});
    z.nodes.push_back(suf::PostprocessBoolNode{suf::PostprocessBoolOp::KAPPA_B, -1, -1, 0, 0});
    z.nodes.push_back(suf::PostprocessBoolNode{suf::PostprocessBoolOp::OR, 0, 1, -1, 0});
    z.root = 2;
    spec.postprocess.bool_exprs.push_back(z);
    spec.postprocess.boolean_outputs = {0};
    return spec;
}

static int runFuseFSSPrimitiveCanary(int argc, char **argv)
{
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0]
                  << " fusefss-primitive-canary <N> <party> <ip> <port> [bw]\n";
        return 2;
    }
    const int N = std::atoi(argv[2]);
    const int party = std::atoi(argv[3]);
    const std::string ip(argv[4]);
    const int port = std::atoi(argv[5]);
    const int bw = (argc > 6) ? std::atoi(argv[6]) : 37;
    const int scale = 12;
    if (N <= 0 || bw <= 0 || bw > 64)
    {
        std::cerr << "invalid N or bw for fusefss-primitive-canary\n";
        return 2;
    }
    const u64 mod_mask = (bw >= 64) ? ~0ULL : ((1ULL << bw) - 1ULL);
    const u64 keybuf_mb = envU64("SUF_PRIMITIVE_KEYBUF_MB", 2048);
    const size_t keybuf_size = keybuf_mb * 1024ULL * 1024ULL;

    initGPURandomness();
    initGPUMemPool();

    std::vector<u64> h_lhs(N), h_rhs(N), h_lhs_mask(N), h_rhs_mask(N);
    std::vector<u64> h_lhs_open(N), h_rhs_open(N);
    std::vector<u64> h_lhs_bit_mask(N), h_rhs_bit_mask(N);
    std::vector<u64> h_lhs_bit_open(N), h_rhs_bit_open(N);
    std::vector<u64> h_relu_x(N), h_relu_mask(N), h_relu_open(N), h_relu_open_share(N);
    std::vector<u64> h_v3_x(N), h_v3_mask(N), h_v3_open(N), h_v3_open_share(N);
    std::vector<u64> h_v3_kappa_a(N), h_v3_kappa_a_mask(N), h_v3_kappa_a_open(N);
    std::vector<u64> h_v3_kappa_b(N), h_v3_kappa_b_mask(N), h_v3_kappa_b_open(N);
    std::vector<u64> h_or_x(N), h_or_mask(N), h_or_open(N), h_or_open_share(N);
    std::vector<u64> h_or_kappa_b(N), h_or_kappa_b_mask(N), h_or_kappa_b_open(N);
    for (int i = 0; i < N; ++i)
    {
        h_lhs[i] = (static_cast<u64>(i) * 1315423911ULL + 17ULL) & mod_mask;
        h_rhs[i] = (static_cast<u64>(i) * 2654435761ULL + 29ULL) & mod_mask;
        h_lhs_mask[i] = (static_cast<u64>(i) * 11400714819323198485ULL + 5ULL) & mod_mask;
        h_rhs_mask[i] = (static_cast<u64>(i) * 7046029254386353131ULL + 11ULL) & mod_mask;
        h_lhs_open[i] = (h_lhs[i] + h_lhs_mask[i]) & mod_mask;
        h_rhs_open[i] = (h_rhs[i] + h_rhs_mask[i]) & mod_mask;
        h_lhs_bit_mask[i] = h_lhs_mask[i] & 1ULL;
        h_rhs_bit_mask[i] = h_rhs_mask[i] & 1ULL;
        h_lhs_bit_open[i] = (h_lhs[i] & 1ULL) ^ h_lhs_bit_mask[i];
        h_rhs_bit_open[i] = (h_rhs[i] & 1ULL) ^ h_rhs_bit_mask[i];
        h_relu_x[i] = (static_cast<u64>(i) * 73ULL + 19ULL) & 0xFFULL;
        h_relu_mask[i] = (static_cast<u64>(i) * 151ULL + 7ULL) & 0xFFULL;
        h_relu_open[i] = (h_relu_x[i] + h_relu_mask[i]) & 0xFFULL;
        h_relu_open_share[i] = (party == SERVER1) ? h_relu_open[i] : 0ULL;
        h_v3_x[i] = (static_cast<u64>(i) * 41ULL + 23ULL) & 0xFFULL;
        h_v3_mask[i] = (static_cast<u64>(i) * 97ULL + 13ULL) & 0xFFULL;
        h_v3_open[i] = (h_v3_x[i] + h_v3_mask[i]) & 0xFFULL;
        h_v3_open_share[i] = (party == SERVER1) ? h_v3_open[i] : 0ULL;
        h_v3_kappa_a[i] = (static_cast<u64>(i) * 5ULL + 9ULL) & 0xFFULL;
        h_v3_kappa_a_mask[i] = (static_cast<u64>(i) * 29ULL + 3ULL) & 0xFFULL;
        h_v3_kappa_a_open[i] = (h_v3_kappa_a[i] + h_v3_kappa_a_mask[i]) & 0xFFULL;
        h_v3_kappa_b[i] = (static_cast<u64>(i >> 1) & 1ULL);
        h_v3_kappa_b_mask[i] = (static_cast<u64>(i >> 2) & 1ULL);
        h_v3_kappa_b_open[i] = h_v3_kappa_b[i] ^ h_v3_kappa_b_mask[i];
        h_or_x[i] = (static_cast<u64>(i) * 59ULL + 31ULL) & 0xFFULL;
        h_or_mask[i] = (static_cast<u64>(i) * 17ULL + 23ULL) & 0xFFULL;
        h_or_open[i] = (h_or_x[i] + h_or_mask[i]) & 0xFFULL;
        h_or_open_share[i] = (party == SERVER1) ? h_or_open[i] : 0ULL;
        h_or_kappa_b[i] = (static_cast<u64>(i >> 3) & 1ULL);
        h_or_kappa_b_mask[i] = (static_cast<u64>(i >> 4) & 1ULL);
        h_or_kappa_b_open[i] = h_or_kappa_b[i] ^ h_or_kappa_b_mask[i];
    }

    const size_t bytes = static_cast<size_t>(N) * sizeof(u64);
    auto *d_lhs_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_lhs_mask.data()), bytes, nullptr));
    auto *d_rhs_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_rhs_mask.data()), bytes, nullptr));
    auto *d_lhs_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_lhs_open.data()), bytes, nullptr));
    auto *d_rhs_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_rhs_open.data()), bytes, nullptr));
    auto *d_lhs_bit_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_lhs_bit_mask.data()), bytes, nullptr));
    auto *d_rhs_bit_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_rhs_bit_mask.data()), bytes, nullptr));
    auto *d_lhs_bit_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_lhs_bit_open.data()), bytes, nullptr));
    auto *d_rhs_bit_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_rhs_bit_open.data()), bytes, nullptr));
    auto *d_relu_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_relu_mask.data()), bytes, nullptr));
    auto *d_relu_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_relu_open_share.data()), bytes, nullptr));
    auto *d_v3_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_v3_mask.data()), bytes, nullptr));
    auto *d_v3_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_v3_open_share.data()), bytes, nullptr));
    auto *d_v3_kappa_a_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_v3_kappa_a_mask.data()), bytes, nullptr));
    auto *d_v3_kappa_a_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_v3_kappa_a_open.data()), bytes, nullptr));
    auto *d_v3_kappa_b_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_v3_kappa_b_mask.data()), bytes, nullptr));
    auto *d_v3_kappa_b_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_v3_kappa_b_open.data()), bytes, nullptr));
    auto *d_or_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_or_mask.data()), bytes, nullptr));
    auto *d_or_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_or_open_share.data()), bytes, nullptr));
    auto *d_or_kappa_b_mask = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_or_kappa_b_mask.data()), bytes, nullptr));
    auto *d_or_kappa_b_open = reinterpret_cast<u64 *>(moveToGPU(reinterpret_cast<u8 *>(h_or_kappa_b_open.data()), bytes, nullptr));

    bool pin_keybuf = true;
    if (const char *pin_env = std::getenv("SIGMA_PINNED_KEYBUF"))
    {
        if (pin_env[0])
            pin_keybuf = std::atoi(pin_env) != 0;
    }
    if (const char *pin_env = fusefssEnvValue("SUF_PRIMITIVE_PINNED_KEYBUF"))
    {
        if (pin_env[0])
            pin_keybuf = std::atoi(pin_env) != 0;
    }
    auto *key_start = cpuMalloc(keybuf_size, pin_keybuf);
    auto *key_ptr = key_start;
    suf_sigma_set_keybuf_ptr(&key_ptr);
    suf_sigma_reset_keygen();

    auto keygen_start = std::chrono::high_resolution_clock::now();
    auto mul_keygen_start = keygen_start;
    auto *d_mul_mask = suf_sigma_keygen_postprocess_mul_u64(party, bw, scale,
                                                            d_lhs_mask, d_rhs_mask,
                                                            static_cast<size_t>(N));
    cudaDeviceSynchronize();
    auto mul_keygen_end = std::chrono::high_resolution_clock::now();
    auto and_keygen_start = mul_keygen_end;
    auto *d_and_mask = suf_sigma_keygen_postprocess_and_u64(party,
                                                            d_lhs_bit_mask, d_rhs_bit_mask,
                                                            static_cast<size_t>(N));
    cudaDeviceSynchronize();
    auto and_keygen_end = std::chrono::high_resolution_clock::now();
    auto b2a_keygen_start = and_keygen_end;
    auto *d_b2a_mask = suf_sigma_keygen_postprocess_b2a_u64(party, bw,
                                                            d_lhs_bit_mask,
                                                            static_cast<size_t>(N));
    cudaDeviceSynchronize();
    auto b2a_keygen_end = std::chrono::high_resolution_clock::now();
    auto a2b_keygen_start = b2a_keygen_end;
    auto *d_a2b_mask = suf_sigma_keygen_postprocess_a2b_lsb_u64(party,
                                                                d_lhs_mask,
                                                                static_cast<size_t>(N));
    cudaDeviceSynchronize();
    auto a2b_keygen_end = std::chrono::high_resolution_clock::now();
    auto relu_keygen_start = a2b_keygen_end;
    auto relu_spec = makeFuseFSSAffineReluPhiSpec();
    const int relu_op = suf_sigma_register_operator_spec(&relu_spec);
    const int relu_scale = 0;
    auto *relu_key = suf_sigma_keygen_compiled_operator_v2(relu_op, party, bw, relu_scale,
                                                           d_relu_mask,
                                                           static_cast<size_t>(N));
    cudaDeviceSynchronize();
    auto relu_keygen_end = std::chrono::high_resolution_clock::now();
    auto v3_keygen_start = relu_keygen_end;
    auto v3_spec = makeFuseFSSKappaA2BBitSpec();
    const int v3_op = suf_sigma_register_operator_spec(&v3_spec);
    const int v3_bw = 8;
    const int v3_scale = 0;
    bool missing_kappa_failfast = false;
    bool short_kappa_failfast = false;
    try {
        auto *bad = suf_sigma_keygen_compiled_operator_v3(v3_op, party, v3_bw, v3_scale,
                                                          d_v3_mask,
                                                          static_cast<size_t>(N),
                                                          nullptr);
        if (bad)
            suf_sigma_free_compiled_operator_result_v2(bad);
    } catch (const std::exception &) {
        missing_kappa_failfast = true;
    }
    const u64 *v3_kappa_a_key_ptrs[] = {d_v3_kappa_a_mask};
    const u64 *v3_kappa_b_key_ptrs[] = {d_v3_kappa_b_mask};
    const std::size_t one_kappa_len[] = {static_cast<std::size_t>(N)};
    const std::size_t short_kappa_len[] = {static_cast<std::size_t>(N - 1)};
    SufSigmaPostprocessContext short_key_ctx{};
    short_key_ctx.kappa_a_count = 1;
    short_key_ctx.d_kappa_a = v3_kappa_a_key_ptrs;
    short_key_ctx.kappa_a_lengths = short_kappa_len;
    short_key_ctx.kappa_b_count = 1;
    short_key_ctx.d_kappa_b = v3_kappa_b_key_ptrs;
    short_key_ctx.kappa_b_lengths = one_kappa_len;
    try {
        auto *bad = suf_sigma_keygen_compiled_operator_v3(v3_op, party, v3_bw, v3_scale,
                                                          d_v3_mask,
                                                          static_cast<size_t>(N),
                                                          &short_key_ctx);
        if (bad)
            suf_sigma_free_compiled_operator_result_v2(bad);
    } catch (const std::exception &) {
        short_kappa_failfast = true;
    }
    SufSigmaPostprocessContext v3_key_ctx{};
    v3_key_ctx.kappa_a_count = 1;
    v3_key_ctx.d_kappa_a = v3_kappa_a_key_ptrs;
    v3_key_ctx.kappa_a_lengths = one_kappa_len;
    v3_key_ctx.kappa_b_count = 1;
    v3_key_ctx.d_kappa_b = v3_kappa_b_key_ptrs;
    v3_key_ctx.kappa_b_lengths = one_kappa_len;
    auto *v3_key = suf_sigma_keygen_compiled_operator_v3(v3_op, party, v3_bw, v3_scale,
                                                         d_v3_mask,
                                                         static_cast<size_t>(N),
                                                         &v3_key_ctx);
    cudaDeviceSynchronize();
    auto v3_keygen_end = std::chrono::high_resolution_clock::now();
    auto or_keygen_start = v3_keygen_end;
    auto or_spec = makeFuseFSSBoolOrSpec();
    const int or_op = suf_sigma_register_operator_spec(&or_spec);
    const u64 *or_kappa_b_key_ptrs[] = {d_or_kappa_b_mask};
    SufSigmaPostprocessContext or_key_ctx{};
    or_key_ctx.kappa_b_count = 1;
    or_key_ctx.d_kappa_b = or_kappa_b_key_ptrs;
    or_key_ctx.kappa_b_lengths = one_kappa_len;
    auto *or_key = suf_sigma_keygen_compiled_operator_v3(or_op, party, v3_bw, v3_scale,
                                                        d_or_mask,
                                                        static_cast<size_t>(N),
                                                        &or_key_ctx);
    cudaDeviceSynchronize();
    auto or_keygen_end = std::chrono::high_resolution_clock::now();
    auto keygen_end = std::chrono::high_resolution_clock::now();
    const size_t key_bytes = static_cast<size_t>(key_ptr - key_start);

    auto peer = new GpuPeer(true);
    peer->connect(party, ip, port);
    key_ptr = key_start;
    suf_sigma_set_keybuf_ptr(&key_ptr);
    suf_sigma_reset_eval();
    Stats mul_stats;
    Stats and_stats;
    peer->sync();
    const u64 comm0 = peer->bytesSent() + peer->bytesReceived();
    auto eval_start = std::chrono::high_resolution_clock::now();
    auto mul_eval_start = eval_start;
    auto *d_mul_open = suf_sigma_eval_postprocess_mul_u64(peer, party, bw, scale,
                                                          d_lhs_open, d_rhs_open,
                                                          static_cast<size_t>(N), &mul_stats);
    cudaDeviceSynchronize();
    auto mul_eval_end = std::chrono::high_resolution_clock::now();
    const u64 comm_after_mul = peer->bytesSent() + peer->bytesReceived();
    auto and_eval_start = mul_eval_end;
    auto *d_and_open = suf_sigma_eval_postprocess_and_u64(peer, party,
                                                          d_lhs_bit_open, d_rhs_bit_open,
                                                          static_cast<size_t>(N), &and_stats);
    cudaDeviceSynchronize();
    auto and_eval_end = std::chrono::high_resolution_clock::now();
    const u64 comm_after_and = peer->bytesSent() + peer->bytesReceived();
    Stats b2a_stats;
    auto b2a_eval_start = and_eval_end;
    auto *d_b2a_open = suf_sigma_eval_postprocess_b2a_u64(peer, party, bw,
                                                          d_lhs_bit_open,
                                                          static_cast<size_t>(N),
                                                          &b2a_stats);
    cudaDeviceSynchronize();
    auto b2a_eval_end = std::chrono::high_resolution_clock::now();
    const u64 comm_after_b2a = peer->bytesSent() + peer->bytesReceived();
    Stats a2b_stats;
    auto a2b_eval_start = b2a_eval_end;
    auto *d_a2b_open = suf_sigma_eval_postprocess_a2b_lsb_u64(peer, party,
                                                              d_lhs_open,
                                                              static_cast<size_t>(N),
                                                              &a2b_stats);
    cudaDeviceSynchronize();
    auto a2b_eval_end = std::chrono::high_resolution_clock::now();
    const u64 comm_after_a2b = peer->bytesSent() + peer->bytesReceived();
    Stats relu_stats;
    auto relu_eval_start = a2b_eval_end;
    auto *relu_eval = suf_sigma_eval_compiled_operator_v2(peer, relu_op, party, bw, relu_scale,
                                                          d_relu_open,
                                                          static_cast<size_t>(N),
                                                          &relu_stats);
    cudaDeviceSynchronize();
    auto relu_eval_end = std::chrono::high_resolution_clock::now();
    const u64 comm_after_relu = peer->bytesSent() + peer->bytesReceived();
    Stats v3_stats;
    auto v3_eval_start = relu_eval_end;
    const u64 *v3_kappa_a_eval_ptrs[] = {d_v3_kappa_a_open};
    const u64 *v3_kappa_b_eval_ptrs[] = {d_v3_kappa_b_open};
    SufSigmaPostprocessContext v3_eval_ctx{};
    v3_eval_ctx.kappa_a_count = 1;
    v3_eval_ctx.d_kappa_a = v3_kappa_a_eval_ptrs;
    v3_eval_ctx.kappa_a_lengths = one_kappa_len;
    v3_eval_ctx.kappa_b_count = 1;
    v3_eval_ctx.d_kappa_b = v3_kappa_b_eval_ptrs;
    v3_eval_ctx.kappa_b_lengths = one_kappa_len;
    auto *v3_eval = suf_sigma_eval_compiled_operator_v3(peer, v3_op, party, v3_bw, v3_scale,
                                                        d_v3_open,
                                                        static_cast<size_t>(N),
                                                        &v3_stats,
                                                        &v3_eval_ctx);
    cudaDeviceSynchronize();
    auto v3_eval_end = std::chrono::high_resolution_clock::now();
    const u64 comm_after_v3 = peer->bytesSent() + peer->bytesReceived();
    Stats or_stats;
    auto or_eval_start = v3_eval_end;
    const u64 *or_kappa_b_eval_ptrs[] = {d_or_kappa_b_open};
    SufSigmaPostprocessContext or_eval_ctx{};
    or_eval_ctx.kappa_b_count = 1;
    or_eval_ctx.d_kappa_b = or_kappa_b_eval_ptrs;
    or_eval_ctx.kappa_b_lengths = one_kappa_len;
    auto *or_eval = suf_sigma_eval_compiled_operator_v3(peer, or_op, party, v3_bw, v3_scale,
                                                       d_or_open,
                                                       static_cast<size_t>(N),
                                                       &or_stats,
                                                       &or_eval_ctx);
    cudaDeviceSynchronize();
    auto or_eval_end = std::chrono::high_resolution_clock::now();
    const u64 comm_after_or = peer->bytesSent() + peer->bytesReceived();
    auto eval_end = std::chrono::high_resolution_clock::now();
    peer->sync();
    const u64 comm1 = peer->bytesSent() + peer->bytesReceived();

    std::vector<u64> h_mul_open(N), h_mul_mask(N), h_and_open(N), h_and_mask(N);
    std::vector<u64> h_b2a_open(N), h_b2a_mask(N), h_a2b_open(N), h_a2b_mask(N);
    std::vector<u64> h_relu_x_open(N), h_relu_x_mask(N);
    std::vector<u64> h_relu_a_open(N), h_relu_a_mask(N), h_relu_b_open(N), h_relu_b_mask(N);
    std::vector<u64> h_relu_prod_open(N), h_relu_prod_mask(N);
    std::vector<u64> h_relu_y_open(N), h_relu_y_mask(N), h_relu_z_open(N), h_relu_z_mask(N);
    std::vector<u64> h_v3_y_open(N), h_v3_y_mask(N), h_v3_z_open(N), h_v3_z_mask(N);
    std::vector<u64> h_or_z_open(N), h_or_z_mask(N);
    cudaMemcpy(h_mul_open.data(), d_mul_open, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mul_mask.data(), d_mul_mask, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_and_open.data(), d_and_open, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_and_mask.data(), d_and_mask, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2a_open.data(), d_b2a_open, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2a_mask.data(), d_b2a_mask, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a2b_open.data(), d_a2b_open, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a2b_mask.data(), d_a2b_mask, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_x_open.data(), relu_eval->d_arithmetic[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_x_mask.data(), relu_key->d_arithmetic[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_a_open.data(), relu_eval->d_arithmetic[1], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_a_mask.data(), relu_key->d_arithmetic[1], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_b_open.data(), relu_eval->d_arithmetic[2], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_b_mask.data(), relu_key->d_arithmetic[2], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_prod_open.data(), relu_eval->d_arithmetic[3], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_prod_mask.data(), relu_key->d_arithmetic[3], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_y_open.data(), relu_eval->d_arithmetic[4], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_y_mask.data(), relu_key->d_arithmetic[4], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_z_open.data(), relu_eval->d_boolean[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_relu_z_mask.data(), relu_key->d_boolean[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v3_y_open.data(), v3_eval->d_arithmetic[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v3_y_mask.data(), v3_key->d_arithmetic[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v3_z_open.data(), v3_eval->d_boolean[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v3_z_mask.data(), v3_key->d_boolean[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_or_z_open.data(), or_eval->d_boolean[0], bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_or_z_mask.data(), or_key->d_boolean[0], bytes, cudaMemcpyDeviceToHost);

    bool verified = true;
    int first_bad = -1;
    bool bad_mul = false;
    bool bad_and = false;
    bool bad_b2a = false;
    bool bad_a2b = false;
    bool bad_relu_x = false;
    bool bad_relu_a = false;
    bool bad_relu_b = false;
    bool bad_relu_prod = false;
    bool bad_relu_y = false;
    bool bad_relu_z = false;
    bool bad_v3_y = false;
    bool bad_v3_z = false;
    bool bad_or_z = false;
    if (!missing_kappa_failfast || !short_kappa_failfast)
        verified = false;
    u64 first_got_mul = 0, first_exp_mul = 0;
    u64 first_got_and = 0, first_exp_and = 0;
    u64 first_got_b2a = 0, first_exp_b2a = 0;
    u64 first_got_a2b = 0, first_exp_a2b = 0;
    u64 first_got_relu_x = 0, first_exp_relu_x = 0;
    u64 first_got_relu_a = 0, first_exp_relu_a = 0;
    u64 first_got_relu_b = 0, first_exp_relu_b = 0;
    u64 first_raw_relu_x_open = 0, first_raw_relu_x_mask = 0;
    u64 first_raw_relu_a_open = 0, first_raw_relu_a_mask = 0;
    u64 first_raw_relu_b_open = 0, first_raw_relu_b_mask = 0;
    u64 first_got_relu_prod = 0, first_exp_relu_prod = 0;
    u64 first_got_relu_y = 0, first_exp_relu_y = 0;
    u64 first_got_relu_z = 0, first_exp_relu_z = 0;
    u64 first_got_v3_y = 0, first_exp_v3_y = 0;
    u64 first_got_v3_z = 0, first_exp_v3_z = 0;
    u64 first_got_or_z = 0, first_exp_or_z = 0;
    for (int i = 0; i < N; ++i)
    {
        const u64 got_mul = (h_mul_open[i] - h_mul_mask[i]) & mod_mask;
        const u64 exp_mul = static_cast<u64>(
            static_cast<unsigned __int128>(h_lhs[i]) *
            static_cast<unsigned __int128>(h_rhs[i])) & mod_mask;
        const u64 got_and = (h_and_open[i] ^ h_and_mask[i]) & 1ULL;
        const u64 exp_and = (h_lhs[i] & h_rhs[i]) & 1ULL;
        const u64 got_b2a = (h_b2a_open[i] - h_b2a_mask[i]) & mod_mask;
        const u64 exp_b2a = h_lhs[i] & 1ULL;
        const u64 got_a2b = (h_a2b_open[i] ^ h_a2b_mask[i]) & 1ULL;
        const u64 exp_a2b = h_lhs[i] & 1ULL;
        const u64 got_relu_x = (h_relu_x_open[i] - h_relu_x_mask[i]) & mod_mask;
        const u64 exp_relu_x = h_relu_x[i] & mod_mask;
        const bool relu_positive = h_relu_x[i] < 128ULL;
        const u64 got_relu_a = (h_relu_a_open[i] - h_relu_a_mask[i]) & mod_mask;
        const u64 exp_relu_a = relu_positive ? 1ULL : 0ULL;
        const u64 got_relu_b = (h_relu_b_open[i] - h_relu_b_mask[i]) & mod_mask;
        const u64 exp_relu_b = relu_positive ? 3ULL : 5ULL;
        const u64 got_relu_prod = (h_relu_prod_open[i] - h_relu_prod_mask[i]) & mod_mask;
        const u64 exp_relu_prod = relu_positive ? h_relu_x[i] : 0ULL;
        const u64 got_relu_y = (h_relu_y_open[i] - h_relu_y_mask[i]) & mod_mask;
        const u64 exp_relu_y = (h_relu_x[i] < 128ULL) ? ((h_relu_x[i] + 3ULL) & mod_mask)
                                                       : (5ULL & mod_mask);
        const u64 got_relu_z = (h_relu_z_open[i] ^ h_relu_z_mask[i]) & 1ULL;
        const u64 exp_relu_z = (h_relu_x[i] >= 128ULL) ? 1ULL : 0ULL;
        const u64 got_v3_y = (h_v3_y_open[i] - h_v3_y_mask[i]) & 0xFFULL;
        const u64 exp_v3_y = (1ULL + h_v3_kappa_a[i]) & 0xFFULL;
        const u64 got_v3_z = (h_v3_z_open[i] ^ h_v3_z_mask[i]) & 1ULL;
        const u64 exp_v3_z = (h_v3_kappa_b[i] ^ ((h_v3_x[i] >> 3) & 1ULL)) & 1ULL;
        const u64 got_or_z = (h_or_z_open[i] ^ h_or_z_mask[i]) & 1ULL;
        const u64 exp_or_z = ((h_or_x[i] >= 128ULL) ? 1ULL : 0ULL) | h_or_kappa_b[i];
        if (got_mul != exp_mul || got_and != exp_and ||
            got_b2a != exp_b2a || got_a2b != exp_a2b ||
            got_relu_x != exp_relu_x || got_relu_a != exp_relu_a ||
            got_relu_b != exp_relu_b || got_relu_prod != exp_relu_prod ||
            got_relu_y != exp_relu_y || got_relu_z != exp_relu_z ||
            got_v3_y != exp_v3_y || got_v3_z != exp_v3_z ||
            got_or_z != exp_or_z)
        {
            verified = false;
            first_bad = i;
            bad_mul = got_mul != exp_mul;
            bad_and = got_and != exp_and;
            bad_b2a = got_b2a != exp_b2a;
            bad_a2b = got_a2b != exp_a2b;
            bad_relu_x = got_relu_x != exp_relu_x;
            bad_relu_a = got_relu_a != exp_relu_a;
            bad_relu_b = got_relu_b != exp_relu_b;
            bad_relu_prod = got_relu_prod != exp_relu_prod;
            bad_relu_y = got_relu_y != exp_relu_y;
            bad_relu_z = got_relu_z != exp_relu_z;
            bad_v3_y = got_v3_y != exp_v3_y;
            bad_v3_z = got_v3_z != exp_v3_z;
            bad_or_z = got_or_z != exp_or_z;
            first_got_mul = got_mul;
            first_exp_mul = exp_mul;
            first_got_and = got_and;
            first_exp_and = exp_and;
            first_got_b2a = got_b2a;
            first_exp_b2a = exp_b2a;
            first_got_a2b = got_a2b;
            first_exp_a2b = exp_a2b;
            first_got_relu_x = got_relu_x;
            first_exp_relu_x = exp_relu_x;
            first_got_relu_a = got_relu_a;
            first_exp_relu_a = exp_relu_a;
            first_got_relu_b = got_relu_b;
            first_exp_relu_b = exp_relu_b;
            first_raw_relu_x_open = h_relu_x_open[i];
            first_raw_relu_x_mask = h_relu_x_mask[i];
            first_raw_relu_a_open = h_relu_a_open[i];
            first_raw_relu_a_mask = h_relu_a_mask[i];
            first_raw_relu_b_open = h_relu_b_open[i];
            first_raw_relu_b_mask = h_relu_b_mask[i];
            first_got_relu_prod = got_relu_prod;
            first_exp_relu_prod = exp_relu_prod;
            first_got_relu_y = got_relu_y;
            first_exp_relu_y = exp_relu_y;
            first_got_relu_z = got_relu_z;
            first_exp_relu_z = exp_relu_z;
            first_got_v3_y = got_v3_y;
            first_exp_v3_y = exp_v3_y;
            first_got_v3_z = got_v3_z;
            first_exp_v3_z = exp_v3_z;
            first_got_or_z = got_or_z;
            first_exp_or_z = exp_or_z;
            break;
        }
    }

    const auto keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        keygen_end - keygen_start).count();
    const auto mul_keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        mul_keygen_end - mul_keygen_start).count();
    const auto and_keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        and_keygen_end - and_keygen_start).count();
    const auto b2a_keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        b2a_keygen_end - b2a_keygen_start).count();
    const auto a2b_keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        a2b_keygen_end - a2b_keygen_start).count();
    const auto relu_keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        relu_keygen_end - relu_keygen_start).count();
    const auto v3_keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        v3_keygen_end - v3_keygen_start).count();
    const auto or_keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(
        or_keygen_end - or_keygen_start).count();
    const auto eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        eval_end - eval_start).count();
    const auto mul_eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        mul_eval_end - mul_eval_start).count();
    const auto and_eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        and_eval_end - and_eval_start).count();
    const auto b2a_eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        b2a_eval_end - b2a_eval_start).count();
    const auto a2b_eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        a2b_eval_end - a2b_eval_start).count();
    const auto relu_eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        relu_eval_end - relu_eval_start).count();
    const auto v3_eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        v3_eval_end - v3_eval_start).count();
    const auto or_eval_us = std::chrono::duration_cast<std::chrono::microseconds>(
        or_eval_end - or_eval_start).count();
    std::cout << "{"
              << "\"bench\":\"fusefss_postprocess_primitives\","
              << "\"n\":" << N << ","
              << "\"party\":" << party << ","
              << "\"bw\":" << bw << ","
              << "\"mul_supported\":" << (suf_sigma_postprocess_mul_supported() ? "true" : "false") << ","
              << "\"and_supported\":" << (suf_sigma_postprocess_and_supported() ? "true" : "false") << ","
              << "\"b2a_supported\":" << (suf_sigma_postprocess_b2a_supported() ? "true" : "false") << ","
              << "\"a2b_supported\":" << (suf_sigma_postprocess_a2b_supported() ? "true" : "false") << ","
              << "\"strict_relu_supported\":" << (suf_sigma_compiled_operator_strict_supported(relu_op) ? "true" : "false") << ","
              << "\"strict_v3_supported\":" << (suf_sigma_compiled_operator_strict_supported(v3_op) ? "true" : "false") << ","
              << "\"strict_or_supported\":" << (suf_sigma_compiled_operator_strict_supported(or_op) ? "true" : "false") << ","
              << "\"missing_kappa_failfast\":" << (missing_kappa_failfast ? "true" : "false") << ","
              << "\"short_kappa_failfast\":" << (short_kappa_failfast ? "true" : "false") << ","
              << "\"key_bytes\":" << key_bytes << ","
              << "\"keygen_us\":" << keygen_us << ","
              << "\"mul_keygen_us\":" << mul_keygen_us << ","
              << "\"and_keygen_us\":" << and_keygen_us << ","
              << "\"b2a_keygen_us\":" << b2a_keygen_us << ","
              << "\"a2b_keygen_us\":" << a2b_keygen_us << ","
              << "\"strict_relu_keygen_us\":" << relu_keygen_us << ","
              << "\"strict_v3_keygen_us\":" << v3_keygen_us << ","
              << "\"strict_or_keygen_us\":" << or_keygen_us << ","
              << "\"eval_us\":" << eval_us << ","
              << "\"mul_eval_us\":" << mul_eval_us << ","
              << "\"and_eval_us\":" << and_eval_us << ","
              << "\"b2a_eval_us\":" << b2a_eval_us << ","
              << "\"a2b_eval_us\":" << a2b_eval_us << ","
              << "\"strict_relu_eval_us\":" << relu_eval_us << ","
              << "\"strict_v3_eval_us\":" << v3_eval_us << ","
              << "\"strict_or_eval_us\":" << or_eval_us << ","
              << "\"comm_bytes\":" << (comm1 - comm0) << ","
              << "\"mul_comm_bytes\":" << (comm_after_mul - comm0) << ","
              << "\"and_comm_bytes\":" << (comm_after_and - comm_after_mul) << ","
              << "\"b2a_comm_bytes\":" << (comm_after_b2a - comm_after_and) << ","
              << "\"a2b_comm_bytes\":" << (comm_after_a2b - comm_after_b2a) << ","
              << "\"strict_relu_comm_bytes\":" << (comm_after_relu - comm_after_a2b) << ","
              << "\"strict_v3_comm_bytes\":" << (comm_after_v3 - comm_after_relu) << ","
              << "\"strict_or_comm_bytes\":" << (comm_after_or - comm_after_v3) << ","
              << "\"linear_comm_bytes\":" << (mul_stats.linear_comm_bytes + and_stats.linear_comm_bytes + b2a_stats.linear_comm_bytes + a2b_stats.linear_comm_bytes + relu_stats.linear_comm_bytes + v3_stats.linear_comm_bytes + or_stats.linear_comm_bytes) << ","
              << "\"comm_time_us\":" << (mul_stats.comm_time + and_stats.comm_time + b2a_stats.comm_time + a2b_stats.comm_time + relu_stats.comm_time + v3_stats.comm_time + or_stats.comm_time) << ","
              << "\"transfer_time_us\":" << (mul_stats.transfer_time + and_stats.transfer_time + b2a_stats.transfer_time + a2b_stats.transfer_time + relu_stats.transfer_time + v3_stats.transfer_time + or_stats.transfer_time) << ","
              << "\"mul_comm_time_us\":" << mul_stats.comm_time << ","
              << "\"and_comm_time_us\":" << and_stats.comm_time << ","
              << "\"verified\":" << (verified ? "true" : "false") << ","
              << "\"first_bad\":" << first_bad << ","
              << "\"bad_mul\":" << (bad_mul ? "true" : "false") << ","
              << "\"bad_and\":" << (bad_and ? "true" : "false") << ","
              << "\"bad_b2a\":" << (bad_b2a ? "true" : "false") << ","
              << "\"bad_a2b\":" << (bad_a2b ? "true" : "false") << ","
              << "\"bad_relu_x\":" << (bad_relu_x ? "true" : "false") << ","
              << "\"bad_relu_a\":" << (bad_relu_a ? "true" : "false") << ","
              << "\"bad_relu_b\":" << (bad_relu_b ? "true" : "false") << ","
              << "\"bad_relu_prod\":" << (bad_relu_prod ? "true" : "false") << ","
              << "\"bad_relu_y\":" << (bad_relu_y ? "true" : "false") << ","
              << "\"bad_relu_z\":" << (bad_relu_z ? "true" : "false") << ","
              << "\"bad_v3_y\":" << (bad_v3_y ? "true" : "false") << ","
              << "\"bad_v3_z\":" << (bad_v3_z ? "true" : "false") << ","
              << "\"bad_or_z\":" << (bad_or_z ? "true" : "false") << ","
              << "\"first_got_mul\":" << first_got_mul << ","
              << "\"first_exp_mul\":" << first_exp_mul << ","
              << "\"first_got_and\":" << first_got_and << ","
              << "\"first_exp_and\":" << first_exp_and << ","
              << "\"first_got_b2a\":" << first_got_b2a << ","
              << "\"first_exp_b2a\":" << first_exp_b2a << ","
              << "\"first_got_a2b\":" << first_got_a2b << ","
              << "\"first_exp_a2b\":" << first_exp_a2b << ","
              << "\"first_got_relu_x\":" << first_got_relu_x << ","
              << "\"first_exp_relu_x\":" << first_exp_relu_x << ","
              << "\"first_got_relu_a\":" << first_got_relu_a << ","
              << "\"first_exp_relu_a\":" << first_exp_relu_a << ","
              << "\"first_got_relu_b\":" << first_got_relu_b << ","
              << "\"first_exp_relu_b\":" << first_exp_relu_b << ","
              << "\"first_raw_relu_x_open\":" << first_raw_relu_x_open << ","
              << "\"first_raw_relu_x_mask\":" << first_raw_relu_x_mask << ","
              << "\"first_raw_relu_a_open\":" << first_raw_relu_a_open << ","
              << "\"first_raw_relu_a_mask\":" << first_raw_relu_a_mask << ","
              << "\"first_raw_relu_b_open\":" << first_raw_relu_b_open << ","
              << "\"first_raw_relu_b_mask\":" << first_raw_relu_b_mask << ","
              << "\"first_got_relu_prod\":" << first_got_relu_prod << ","
              << "\"first_exp_relu_prod\":" << first_exp_relu_prod << ","
              << "\"first_got_relu_y\":" << first_got_relu_y << ","
              << "\"first_exp_relu_y\":" << first_exp_relu_y << ","
              << "\"first_got_relu_z\":" << first_got_relu_z << ","
              << "\"first_exp_relu_z\":" << first_exp_relu_z << ","
              << "\"first_got_v3_y\":" << first_got_v3_y << ","
              << "\"first_exp_v3_y\":" << first_exp_v3_y << ","
              << "\"first_got_v3_z\":" << first_got_v3_z << ","
              << "\"first_exp_v3_z\":" << first_exp_v3_z << ","
              << "\"first_got_or_z\":" << first_got_or_z << ","
              << "\"first_exp_or_z\":" << first_exp_or_z
              << "}" << std::endl;

    suf_sigma_free_compiled_operator_result_v2(relu_key);
    suf_sigma_free_compiled_operator_result_v2(relu_eval);
    suf_sigma_free_compiled_operator_result_v2(v3_key);
    suf_sigma_free_compiled_operator_result_v2(v3_eval);
    suf_sigma_free_compiled_operator_result_v2(or_key);
    suf_sigma_free_compiled_operator_result_v2(or_eval);
    peer->close();
    suf_sigma_clear();
    return verified ? 0 : 1;
}
#endif

static int runFuseFSSPrivateModelFailfastCanary()
{
    setenv("SIGMA_MODEL_WEIGHTS_PRIVATE", "1", 1);
    bool threw = false;
    try
    {
        sigmaRejectPublicWeightPathIfPrivate("canary public-weight op");
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    std::cout << "{"
              << "\"bench\":\"fusefss_private_model_failfast\","
              << "\"requested_private_model\":true,"
              << "\"public_weight_path_rejected\":" << (threw ? "true" : "false")
              << "}" << std::endl;
    return threw ? 0 : 1;
}

int main(int __argc, char **__argv)
{
    sytorch_init();

    if (__argc >= 2 &&
        (std::string(__argv[1]) == "fusefss-private-model-failfast-canary" ||
         std::string(__argv[1]) == "suf-private-model-failfast-canary"))
    {
        return runFuseFSSPrivateModelFailfastCanary();
    }

    if (__argc >= 2 &&
        (std::string(__argv[1]) == "fusefss-primitive-canary" ||
         std::string(__argv[1]) == "suf-primitive-canary"))
    {
#ifdef SUF_HAVE_CUDA
        return runFuseFSSPrimitiveCanary(__argc, __argv);
#else
        std::cerr << "fusefss-primitive-canary requires SUF_HAVE_CUDA build\n";
        return 2;
#endif
    }

    u64 n_embd = 0;
    u64 n_head = 0;
    u64 n_layer = 0;
    std::string attnMask = "none";
    std::string qkvFormat = "qkvconcat";
    int bw = 0;
    u64 scale = 12;
    u64 n_seq = atoi(__argv[2]);
    int party = atoi(__argv[3]);
    u64 batch = envU64("SIGMA_BATCH", 1);
    if (batch < 1)
        batch = 1;
    const bool random_weights = envFlag("SIGMA_RANDOM_WEIGHTS");
    const bool private_model_weights = envFlag("SIGMA_MODEL_WEIGHTS_PRIVATE") || envFlag("SUF_MODEL_PRIVATE");
    const bool dump_output = envFlag("SIGMA_DUMP_OUTPUT");
    const bool clear_ref = envFlag("SIGMA_CLEAR_REF");
    if (private_model_weights)
    {
        fprintf(stderr,
                "SIGMA_MODEL_WEIGHTS_PRIVATE/FUSEFSS_MODEL_PRIVATE requested: the backend will fail-fast on any public-weight matmul/MHA/layernorm path.\n");
    }

    std::string model(__argv[1]);
    printf("Model=%s\n", model.data());
    u64 keyBufSz = 0;
    SytorchModule<u64> *net;
    Tensor<u64> input({n_seq, n_embd});

    if (model == "gpt2")
    {
        n_layer = 12;
        n_head = 12;
        n_embd = 768;
        attnMask = "self";
        bw = 50;
        u64 mul = (u64)std::pow(2.3, std::log2(n_seq / 64));
        keyBufSz = 10 * mul * OneGB;
        net = new GPUGPT2<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "bert-tiny")
    {
        n_layer = 2;
        n_head = 2;
        n_embd = 128;
        bw = 37;
        keyBufSz = OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "bert-base")
    {
        n_layer = 12;
        n_head = 12;
        n_embd = 768;
        bw = 50;
        keyBufSz = 20 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "bert-large")
    {
        n_layer = 24;
        n_head = 16;
        n_embd = 1024;
        bw = 50;
        keyBufSz = 50 * OneGB;
        net = new GPUBERT<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "gpt-neo")
    {
        n_layer = 24;
        n_head = 16;
        n_embd = 2048;
        attnMask = "self";
        qkvFormat = "kvqsep";
        bw = 51;
        keyBufSz = 80 * OneGB;
        net = new GPUGPT2<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat, false);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "gpt-neo-large")
    {
        n_layer = 32;
        n_head = 20;
        n_embd = 2560;
        attnMask = "self";
        qkvFormat = "concat";
        bw = 51; // 52;
        keyBufSz = 200 * OneGB;
        net = new GPUGPT2<u64>(n_layer, n_head, n_embd, attnMask, qkvFormat, false);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "llama7b")
    {
        n_layer = 32;
        n_head = 32;
        n_embd = 4096;
        attnMask = "self";
        qkvFormat = "qkvsep";
        bw = 48;
        u64 intermediate_size = 11008;
        keyBufSz = 300 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "llama13b")
    {
        n_layer = 40;
        n_head = 40;
        n_embd = 5120;
        attnMask = "self";
        qkvFormat = "qkvsep";
        bw = 48;
        u64 intermediate_size = 13824;
        keyBufSz = 450 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "llama3-8b")
    {
        n_layer = 32;
        n_head = 32;
        n_embd = 4096;
        attnMask = "self";
        qkvFormat = "qkvsep";
        bw = 50;
        u64 intermediate_size = 14336;
        keyBufSz = 300 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    else if (model == "qwen7b")
    {
        n_layer = 28;
        n_head = 28;
        n_embd = 3584;
        attnMask = "self";
        qkvFormat = "qkvsep";
        bw = 50;
        u64 intermediate_size = 18944;
        keyBufSz = 300 * OneGB;
        net = new GPULlama<u64>(n_layer, n_head, n_embd, intermediate_size);
        input.resize({n_seq, n_embd});
        input.zero();
        net->init(scale, input);
        if (random_weights)
            net->randomize();
        else
            net->zero();
    }
    bool keybuf_override = false;
    const char *keybuf_mb = std::getenv("SIGMA_KEYBUF_MB");
    const char *keybuf_gb = std::getenv("SIGMA_KEYBUF_GB");
    if (keybuf_mb && keybuf_mb[0]) {
        keyBufSz = std::strtoull(keybuf_mb, nullptr, 10) * 1024ULL * 1024ULL;
        keybuf_override = true;
    } else if (keybuf_gb && keybuf_gb[0]) {
        keyBufSz = std::strtoull(keybuf_gb, nullptr, 10) * 1024ULL * 1024ULL * 1024ULL;
        keybuf_override = true;
    }
    if (!keybuf_override && batch > 1) {
        keyBufSz *= batch;
    }
    printf("KeyBufSz=%s\n", toGB(keyBufSz).c_str());
    srand(time(NULL));
    std::string outDir = "output/P" + std::to_string(party) + "/models/";
    std::filesystem::create_directories(outDir);
    auto inferenceDir = outDir + model + "-" + std::to_string(n_seq);
    if (batch > 1)
        inferenceDir += "-b" + std::to_string(batch);
    inferenceDir += "/";
    std::filesystem::create_directories(inferenceDir);

    if (clear_ref)
    {
        fprintf(stderr, "SIGMA_CLEAR_REF requested, but ClearText backend lacks MHA support; skipping.\n");
    }

    auto sigmaKeygen = new SIGMAKeygen<u64>(party, bw, scale, "", keyBufSz);
    net->setBackend(sigmaKeygen);
    net->optimize();
    auto start = std::chrono::high_resolution_clock::now();
    const size_t input_bytes = input.size() * sizeof(u64);
    input.d_data = (u64 *)moveToGPU((u8 *)input.data, input_bytes, (Stats *)NULL);
    for (u64 i = 0; i < batch; ++i)
    {
        if (i > 0)
            moveIntoGPUMem((u8 *)input.d_data, (u8 *)input.data, input_bytes, (Stats *)NULL);
        auto &activation = net->forward(input);
        sigmaKeygen->output(activation);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    sigmaKeygen->close();
    std::stringstream ss;
    u64 total_us = static_cast<u64>(elapsed.count());
    u64 per_us = total_us / batch;
    ss << "Batch=" + std::to_string(batch);
    ss << std::endl;
    ss << "Total time=" + std::to_string(total_us) + " us";
    ss << std::endl;
    ss << "Per-inference time=" + std::to_string(per_us) + " us";
    ss << std::endl;
    ss << "Key size=" + toGB(sigmaKeygen->keySize);
    ss << std::endl;
    std::ofstream statsFile(inferenceDir + "dealer.txt");
    statsFile << ss.rdbuf();
    statsFile.close();

    std::string ip(__argv[4]);
    auto sigma = new SIGMA<u64>(party, ip, "", bw, scale, n_seq, n_embd, atoi(__argv[5]), false);
    sigma->keyBuf = sigmaKeygen->startPtr;
    sigma->startPtr = sigma->keyBuf;
    sigma->keySize = sigmaKeygen->keySize;
    net->setBackend(sigma);
    sigma->peer->sync();
    start = std::chrono::high_resolution_clock::now();
    input.d_data = (u64 *)moveToGPU((u8 *)input.data, input_bytes, (Stats *)NULL);
    Tensor<u64> *activation_ptr = nullptr;
    for (u64 i = 0; i < batch; ++i)
    {
        if (i > 0)
            moveIntoGPUMem((u8 *)input.d_data, (u8 *)input.data, input_bytes, (Stats *)NULL);
        auto &activation = net->forward(input);
        activation_ptr = &activation;
        sigma->output(activation);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    sigma->close();
    if (dump_output)
    {
        if (activation_ptr)
        {
            writeTensorBin(inferenceDir + "output.bin", *activation_ptr);
            writeTensorMeta(inferenceDir + "output_meta.json", *activation_ptr, model, scale, bw);
        }
    }
    auto &activation = *activation_ptr;
    auto signedAct = Tensor<i64>((i64 *)activation.data, activation.shape).as_2d();
    // print(signedAct.as_nd(), scale, (u64) bw);
    auto maxIdx = signedAct.argmax(0);
    printf("%d, %ld\n", maxIdx, activation.data[maxIdx]);

    ss.clear();

    total_us = static_cast<u64>(elapsed.count());
    per_us = total_us / batch;
    ss << "Batch=" + std::to_string(batch);
    ss << std::endl;
    ss << "Total time=" + std::to_string(total_us) + " us";
    ss << std::endl;
    ss << "Per-inference time=" + std::to_string(per_us) + " us";
    ss << std::endl;
    ss << "Comm time=" + std::to_string(sigma->s.comm_time) + " us";
    ss << std::endl;
    ss << "Transfer time=" + std::to_string(sigma->s.transfer_time) + " us";
    ss << std::endl;
    ss << "MHA time=" + std::to_string(sigma->s.mha_time) + " us";
    ss << std::endl;
    ss << "Matmul time=" + std::to_string(sigma->s.matmul_time) + " us";
    ss << std::endl;
    ss << "Truncate time=" + std::to_string(sigma->s.truncate_time) + " us";
    ss << std::endl;
    ss << "Gelu time=" + std::to_string(sigma->s.gelu_time) + " us";
    ss << std::endl;
    ss << "Softmax time=" + std::to_string(sigma->s.softmax_time) + " us";
    ss << std::endl;
    ss << "Layernorm time=" + std::to_string(sigma->s.layernorm_time) + " us";
    ss << std::endl;
    ss << std::endl;
    u64 total_comm_bytes = sigma->peer->bytesSent() + sigma->peer->bytesReceived();
    u64 per_comm_bytes = total_comm_bytes / batch;
    ss << "Total Comm=" + toGB(total_comm_bytes);
    ss << std::endl;
    ss << "Per-inference Comm=" + toGB(per_comm_bytes);
    ss << std::endl;
    ss << "Gelu Comm=" + toGB(sigma->s.gelu_comm_bytes);
    ss << std::endl;
    ss << "Softmax Comm=" + toGB(sigma->s.softmax_comm_bytes);
    ss << std::endl;
    ss << "Layernorm Comm=" + toGB(sigma->s.layernorm_comm_bytes);
    ss << std::endl;

    statsFile.open(inferenceDir + "evaluator.txt");
    statsFile << ss.rdbuf();
    statsFile.close();
    return 0;
}
