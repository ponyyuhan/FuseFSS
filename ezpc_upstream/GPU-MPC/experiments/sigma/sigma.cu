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
#include <filesystem>
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
#include <cstdlib>
#include "gpt2.h"
#include "bert.h"
#include "llama2.h"
#include "backend/sigma.h"

inline std::string toGB(u64 bytes)
{
    return std::to_string(bytes) + " B (" + std::to_string((float)bytes / (1024.0f * 1024.0f * 1024.0f)) + " GB)";
}

inline u64 envU64(const char *name, u64 fallback)
{
    const char *v = std::getenv(name);
    if (!v || !v[0])
        return fallback;
    return std::strtoull(v, nullptr, 10);
}

int main(int __argc, char **__argv)
{
    sytorch_init();

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
    sigma->resetKeyDebug();
    sigma->debugKey("online-start");
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
