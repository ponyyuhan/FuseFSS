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

#include <cassert>
#include <cstdlib>

#include <sytorch/tensor.h>
#include <sytorch/backend/cleartext.h>

#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/misc_utils.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_random.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_gelu.h"

using T = u64;

int main(int argc, char *argv[])
{
    // initCommBufs(true);
    initGPUMemPool();
    AESGlobalContext g;
    initAESContext(&g);
    int bw = 48;//51;
    int bin = 36;//38;
    int scale = 12;
    auto d_reluSubSilu = genLUT<T, reluSubSilu<T>>(10, 6, scale);
    auto ct = new ClearText<i64>();
    ct->bw = bw;

    int N = atoi(argv[3]);
    int party = atoi(argv[1]);

    auto peer = new GpuPeer(true);
    peer->connect(party, argv[2]);

    uint8_t *startPtr, *curPtr;
    size_t keybuf_bytes = 40 * OneGB;
    const char *keybuf_mb = std::getenv("SIGMA_KEYBUF_MB");
    const char *keybuf_gb = std::getenv("SIGMA_KEYBUF_GB");
    if (keybuf_mb && keybuf_mb[0]) {
        keybuf_bytes = std::strtoull(keybuf_mb, nullptr, 10) * (1ULL << 20);
    } else if (keybuf_gb && keybuf_gb[0]) {
        keybuf_bytes = std::strtoull(keybuf_gb, nullptr, 10) * (1ULL << 30);
    }
    getKeyBuf(&startPtr, &curPtr, keybuf_bytes);

    initGPURandomness();
    auto d_mask_X = randomGEOnGpu<T>(N, bw);
    auto h_mask_X = (T *)moveToCPU((u8 *)d_mask_X, N * sizeof(T), NULL);
    T *h_X;
    auto d_masked_X = getMaskedInputOnGpu(N, bw, d_mask_X, &h_X, true, 15);
    const char *skip_verify_env = std::getenv("SIGMA_SKIP_VERIFY");
    const bool skip_verify = (skip_verify_env && skip_verify_env[0] != '\0' && skip_verify_env[0] != '0');

    auto keygen_start = std::chrono::high_resolution_clock::now();
    auto d_mask_O = gpuKeyGenGelu<T, u16, 10>(&curPtr, party, bw, bin, scale, N, d_mask_X, &g);
    auto keygen_end = std::chrono::high_resolution_clock::now();
    auto h_mask_O = (T *)moveToCPU((u8 *)d_mask_O, N * sizeof(T), NULL);
    auto keygen_us = std::chrono::duration_cast<std::chrono::microseconds>(keygen_end - keygen_start).count();
    printf("Keygen time=%lu micros\n", static_cast<unsigned long>(keygen_us));
    printf("Key size=%lu\n", static_cast<unsigned long>(curPtr - startPtr));

    auto k = readGpuGeluKey<T, u16>(&startPtr);
    T *d_O;
    Stats s;
    for (int i = 0; i < 1; i++)
    {
        s.comm_time = 0;
        s.transfer_time = 0;
        peer->sync();
        auto start = std::chrono::high_resolution_clock::now();
        d_O = gpuGelu<T, u16, 10>(peer, party, k, bw, bin, scale, N, d_masked_X, d_reluSubSilu, &g, (Stats *)&s);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        printf("Comm time=%lu micros\n", s.comm_time);
        printf("Transfer time=%lu micros\n", s.transfer_time);
        printf("Silu time=%lu micros\n", std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
        printf("Eval comm bytes=%lu\n", static_cast<unsigned long>(peer->bytesSent() + peer->bytesReceived()));
    }
    unmaskValues(bw, N, d_O, h_mask_O, NULL);
    auto h_O = (T *)moveToCPU((uint8_t *)d_O, N * sizeof(T), (Stats *)NULL);
    gpuFree(d_O);
    destroyGPURandomness();
    if (!skip_verify) {
        Tensor<i64> tIn((i64 *)h_X, {(u64)N});
        Tensor<i64> tOut({(u64)N});
        ct->silu(tIn, tOut, (u64)scale, 0);
        for (int i = 0; i < N; i++)
        {
            if(i < 10) {
                printf("%d=%ld, %ld\n", i, tOut.data[i], h_O[i]);
            }
            if ((u64)tOut.data[i] != h_O[i])
            {
                printf("%d=%ld, %ld, %ld\n", i, tIn.data[i], tOut.data[i], h_O[i]);
                assert(0);
            }
        }
    }
    gpuFree(d_reluSubSilu);
    return 0;
}
