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

#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

#include "utils/gpu_random.h"

#include "gpu_truncate.h"
#include "gpu_window.h"
#include "gpu_lut.h"

inline const char* fusefssLayerNormEnvValue(const char* name)
{
    if (std::strncmp(name, "SUF_", 4) == 0)
    {
        std::string alias = "FUSEFSS_";
        alias += name + 4;
        const char* v = std::getenv(alias.c_str());
        if (v && v[0]) return v;
    }
    return std::getenv(name);
}

#if defined(SUF_HAVE_CUDA)
extern "C" bool suf_layernorm_enabled();
extern "C" void suf_sigma_consume_key();
extern "C" void suf_sigma_consume_rsqrt_key(int bw, int scale, int extradiv, size_t n);
#else
inline bool suf_layernorm_enabled()
{
    const char* v = fusefssLayerNormEnvValue("SUF_LAYERNORM");
    if (v && std::atoi(v) != 0) return true;
    v = fusefssLayerNormEnvValue("SUF_NONLINEAR");
    if (v && std::atoi(v) != 0) return true;
    v = fusefssLayerNormEnvValue("SUF_ACTIVATION");
    return (v && std::atoi(v) != 0);
}
#endif

template <typename T>
struct GPUSqKey
{
    T *a, *c;
};

template <typename T>
GPUSqKey<T> readGPUSqKey(int N, u8** key_as_bytes)
{
    GPUSqKey<T> k;
    u64 memSz = N * sizeof(T);
    k.a = (T*) *key_as_bytes;
    *key_as_bytes += memSz;
    k.c = (T*) *key_as_bytes;
    *key_as_bytes += memSz;
    return k;
}


template <typename T>
struct GPULayerNormKey
{
    GPUTruncateKey<T> muTrKey;
    GPUSqKey<T> sqKey;
    // GPULUTKey<T> invSqrtKey;
    GPUTruncateKey<u16> rsqrtTrKey;
    GPUMulKey<T> wMulKey1;
    GPUMulKey<T> wMulKey2;
};

AvgPoolParams transposeWindow(AvgPoolParams p1) {
    assert(p1.FH == 1 && p1.FW == p1.imgW && p1.strideH == 1 && p1.strideW == p1.imgW);
    AvgPoolParams p2;
    memcpy(&p2, &p1, sizeof(AvgPoolParams));
    p2.FH = p1.imgH;
    p2.FW = 1;
    p2.strideH = p1.imgH;
    p2.strideW = 1;
    initPoolParams(p2);
    return p2;
}

template <typename T>
GPULayerNormKey<T> readGPULayerNormKey(AvgPoolParams p, u8** key_as_bytes, bool computeMu = true)
{
    GPULayerNormKey<T> k;
    auto inSz = getInSz(p);
    auto mSz = getMSz(p);
    if(computeMu) {
        k.muTrKey = readGPUTruncateKey<T>(TruncateType::TrFloor, key_as_bytes);
        // printf("Num Truncations=%d\n", k.muTrKey.N);
    }
    k.sqKey = readGPUSqKey<T>(inSz, key_as_bytes);
    if (suf_layernorm_enabled()) {
        k.rsqrtTrKey = readGPUTruncateKey<u16>(TruncateType::TrWithSlack, key_as_bytes);
        // Skip FuseFSS rsqrt key bytes before reading window-mul keys.
        suf_sigma_consume_rsqrt_key(p.bw, p.scale, p.imgW, mSz);
    }
    // printf("Num sq=%ld\n", inSz);
    // k.invSqrtKey = readGPULUTKey<T>(key_as_bytes);
    k.wMulKey1 = readGPUWindowMulKey<T>(p, TruncateType::TrWithSlack, key_as_bytes);
    // printf("here$$$\n");
    auto p2 = transposeWindow(p);
    k.wMulKey2 = readGPUWindowMulKey<T>(p2, TruncateType::TrWithSlack, key_as_bytes);
    // printf("here$$$\n");
    return k;
}

#include "gpu_layernorm.cu"
