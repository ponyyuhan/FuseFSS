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

#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "helper_cuda.h"
#include "gpu_stats.h"
#include <cassert>

// #include <sys/types.h>

cudaMemPool_t mempool;

static bool mem_debug_enabled()
{
    const char *v = std::getenv("SIGMA_MEM_DEBUG");
    return v && std::atoi(v) != 0;
}

static void log_alloc_attempt(size_t size_in_bytes)
{
    if (!mem_debug_enabled())
        return;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    fprintf(stderr, "[gpu_mem] alloc attempt: %zu bytes (free=%zu total=%zu)\n",
            size_in_bytes, free_bytes, total_bytes);
}

extern "C" void initGPUMemPool()
{
    const char *disable_env = std::getenv("SIGMA_MEMPOOL_DISABLE");
    if (disable_env && std::atoi(disable_env) != 0) {
        return;
    }
    int isMemPoolSupported = 0;
    int device = 0;
    // is it okay to use device=0?
    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
                                           cudaDevAttrMemoryPoolsSupported, device));
    // printf("%d\n", isMemPoolSupported);
    assert(isMemPoolSupported);
    /* implicitly assumes that the device is 0 */

    checkCudaErrors(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    checkCudaErrors(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));
    uint64_t *d_dummy_ptr;
    uint64_t bytes = 0;
    const char *env_mb = std::getenv("SIGMA_MEMPOOL_MB");
    const char *env_gb = std::getenv("SIGMA_MEMPOOL_GB");
    if (env_mb && env_mb[0]) {
        bytes = std::strtoull(env_mb, nullptr, 10) * (1ULL << 20);
    } else if (env_gb && env_gb[0]) {
        bytes = std::strtoull(env_gb, nullptr, 10) * (1ULL << 30);
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (bytes == 0) {
        bytes = std::min<uint64_t>(4ULL << 30, free_bytes / 2);
    }
    if (bytes > free_bytes) {
        bytes = free_bytes / 2;
    }
    checkCudaErrors(cudaMallocAsync(&d_dummy_ptr, bytes, 0));
    checkCudaErrors(cudaFreeAsync(d_dummy_ptr, 0));
    uint64_t reserved_read, threshold_read;
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_read));
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold_read));
    printf("reserved memory: %lu %lu\n", reserved_read, threshold_read);
}

extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    log_alloc_attempt(size_in_bytes);
    cudaError_t err = cudaMallocAsync(&d_a, size_in_bytes, 0);
    if (err != cudaSuccess) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        fprintf(stderr, "[gpu_mem] alloc failed: %zu bytes (free=%zu total=%zu)\n",
                size_in_bytes, free_bytes, total_bytes);
    }
    checkCudaErrors(err);
    return d_a;
}


extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin)
{
    uint8_t *h_a;
    int err = posix_memalign((void **)&h_a, 32, size_in_bytes);
    assert(err == 0 && "posix memalign");
    if (pin)
        checkCudaErrors(cudaHostRegister(h_a, size_in_bytes, cudaHostRegisterDefault));
    return h_a;
}

extern "C" void gpuFree(void *d_a)
{
    checkCudaErrors(cudaFreeAsync(d_a, 0));
}

extern "C" void cpuFree(void *h_a, bool pinned)
{
    if (pinned)
        checkCudaErrors(cudaHostUnregister(h_a));
    free(h_a);
}

extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *h_a = cpuMalloc(size_in_bytes, true);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *d_a = gpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return d_a;
}
