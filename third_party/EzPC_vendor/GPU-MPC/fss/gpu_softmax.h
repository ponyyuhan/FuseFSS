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

#include "gpu_maxpool.h"
#include "gpu_nexp.h"
#include "gpu_lut.h"
#include "gpu_inverse.h"
#include "gpu_window.h"

inline const char* fusefssSoftmaxEnvValue(const char* name)
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
extern "C" bool suf_softmax_enabled();
extern "C" void suf_sigma_consume_key();
extern "C" void suf_sigma_consume_nexp_key(int bw, int scale, size_t n);
extern "C" void suf_sigma_consume_inverse_key(int bw, int scale, int nmax, size_t n);
#else
inline bool suf_softmax_enabled()
{
    const char* v = fusefssSoftmaxEnvValue("SUF_SOFTMAX");
    if (v && std::atoi(v) != 0) return true;
    v = fusefssSoftmaxEnvValue("SUF_NONLINEAR");
    if (v && std::atoi(v) != 0) return true;
    v = fusefssSoftmaxEnvValue("SUF_ACTIVATION");
    return (v && std::atoi(v) != 0);
}
#endif


template <typename T>
struct GPUSoftMaxKey
{
    GPUMaxpoolKey<T> maxPoolKey;
    GPUNExpKey<T> nExpKey;
    GPULUTInverseKey<T> invKey;
    GPUTruncateKey<u16> invTrKey;
    GPUMulKey<T> wMulKey;
};

template <typename T>
GPUSoftMaxKey<T> readGPUSoftMaxKey(MaxpoolParams p, u8 **key_as_bytes)
{
    GPUSoftMaxKey<T> k;
    assert(p.C == 1);
    assert(p.strideH == 1);
    assert(p.strideW == p.FW);

    u64 inSz = getInSz(p);
    u64 mSz = getMSz(p);
    k.maxPoolKey = readGPUMaxpoolKey<T>(p, key_as_bytes);
    if (!suf_softmax_enabled()) {
        k.nExpKey = readGPUNExpKey<T>(key_as_bytes);
        k.invKey = readGPULUTInverseKey<T>(key_as_bytes);
        k.invTrKey = k.invKey.trKey;
    } else {
        // Skip FuseFSS nExp key bytes in the shared key buffer.
        suf_sigma_consume_nexp_key(p.bw, p.scale, inSz);
        k.invTrKey = readGPUTruncateKey<u16>(TruncateType::TrWithSlack, key_as_bytes);
        // Skip FuseFSS inverse key bytes before reading window-mul key.
        suf_sigma_consume_inverse_key(p.bw, p.scale, p.FW, mSz);
    }
    k.wMulKey = readGPUWindowMulKey<T>(p, TruncateType::TrWithSlack, key_as_bytes);
    return k;
}



#include "gpu_softmax.cu"
