#pragma diag_warn 1107

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

#include "clustering_pso_cuda_blocks.hpp"

// PACKET_SIZE deve ser definido no seu projeto (mantive o padrão)
#ifndef PACKET_SIZE
#define PACKET_SIZE 4000.0
#endif

#ifndef MAX_GATEWAYS
#define MAX_GATEWAYS 128   // GARANTIDO SEGURO (se usar 60 gateways)
#endif


// --------------------------- Kernel: init (cada thread inicializa 1 elemento de positions/vel) ---------------------------
__global__ void initParticlesBlockKernel(double* d_positions, double* d_velocities,
    int totalParticles, int numSensors, unsigned long seed)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)totalParticles * (size_t)numSensors;
    if (idx >= total) return;

    // determinístico simples baseado em idx e seed (sem depender de curand aqui)
    unsigned long s = seed ^ (unsigned long)idx;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    double r = (double)(s & 0xFFFF) / (double)0xFFFF;

    d_positions[idx] = r;                 // [0,1)
    d_velocities[idx] = r * 1.0 - 0.5;    // [-0.5,0.5)
}

// --------------------------- Kernel: evaluation (cada thread = 1 parcela/partícula) ---------------------------
// versão otimizada — mantém block-per-swarm + thread-per-particle
// Kernel otimizado com mask bit-packed (uint32 words por gateway)
// Mantém block-per-swarm e thread-per-particle
// Kernel de avaliação usando sensor->gateway adjacency pré-computada
__global__ void evaluateClustersBlockKernel(
    const double* __restrict__ d_positions, // totalParticles * numSensors
    double* __restrict__ d_fitness,         // totalParticles
    int totalParticles,                     // swarmSize
    int numSensors,
    int numGateways,
    const NodeGPU* __restrict__ d_nodes,    // numGateways + numSensors
    const int* __restrict__ d_nextHop,
    const int* __restrict__ d_relaysCount,
    const double* __restrict__ d_clusterRadii,
    const int* __restrict__ d_sensorOffsets, // CSR offsets (numSensors+1)
    const int* __restrict__ d_sensorAdj,     // CSR adjacency
    double bsx, double bsy)
{
    const int GW_LIMIT = 128;
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= totalParticles) return;

    if (numGateways > GW_LIMIT) {
        if (pid == 0) printf("[evaluateClusters] numGateways > %d\n", GW_LIMIT);
        d_fitness[pid] = 0.0;
        return;
    }

    // pointer para vetor desta partícula (numSensors entradas)
    const double* pos = d_positions + (size_t)pid * (size_t)numSensors;

    // ------------------------
    // shared memory for gateways
    // layout: x[numGateways], y[numGateways], E[numGateways]
    // ------------------------
    extern __shared__ double s_buf[]; // expect at least 3 * numGateways * sizeof(double)
    double* s_gw_x = s_buf;
    double* s_gw_y = s_gw_x + numGateways;
    double* s_gw_E = s_gw_y + numGateways;
    // NOTE: we don't need s_r2 here because adjacency is precomputed on host

    // cooperative load of gateway info
    for (int g = threadIdx.x; g < numGateways; g += blockDim.x) {
        const NodeGPU& gw = d_nodes[g];
        s_gw_x[g] = gw.x;
        s_gw_y[g] = gw.y;
        s_gw_E[g] = gw.energy;
    }
    __syncthreads();

    // clusterSizes per-thread (small footprint)
    // assume numGateways <= GW_LIMIT
    uint16_t clusterSizes[GW_LIMIT];
#pragma unroll 4
    for (int g = 0; g < numGateways; ++g) clusterSizes[g] = 0;

    // === build clusters using CSR adjacency (host-built) ===
    // For each sensor, read its candidate gateways from d_sensorOffsets/d_sensorAdj
    for (int s = 0; s < numSensors; ++s) {
        double val = pos[s];

        int start = d_sensorOffsets[s];
        int end = d_sensorOffsets[s + 1];
        int deg = end - start;
        if (deg <= 0) continue;

        // robust pick in [0, deg-1]
        int pick = (int)(val * (double)deg);
        if (pick < 0) pick = 0;
        if (pick >= deg) pick = deg - 1;

        int chosenGateway = d_sensorAdj[start + pick];
        if (chosenGateway >= 0 && chosenGateway < numGateways) {
            clusterSizes[chosenGateway]++;
        }
    }

    // -------------------------------------------------------------
    // Compute lifetime using clusterSizes (per-thread), gateways in shared
    // -------------------------------------------------------------
    const double E_elec = 50e-9;
    const double E_amp  = 10e-12;
    const double E_rx   = E_elec * PACKET_SIZE;
    const double E_agg  = 5e-9 * PACKET_SIZE;

    double minLifetime = 1e300;
    bool anyValid = false;

    for (int g = 0; g < numGateways; ++g) {
        double gwx = s_gw_x[g];
        double gwy = s_gw_y[g];
        double energy = s_gw_E[g];

        double e_intra = (double)clusterSizes[g] * (E_rx + E_agg);

        int nh = d_nextHop[g];
        double e_inter = 0.0;

        if (nh == -1) {
            // gateway -> BS
            double dx = gwx - bsx;
            double dy = gwy - bsy;
            double d = sqrt(dx * dx + dy * dy);           // use sqrt to match thread/CPU
            double Etx = (E_elec + E_amp * d * d) * PACKET_SIZE;
            e_inter = Etx;
        }
        else if (nh >= 0 && nh < numGateways) {
            // gateway -> next gateway (use shared coords)
            double nx = s_gw_x[nh];
            double ny = s_gw_y[nh];
            double dx = gwx - nx;
            double dy = gwy - ny;
            double d = sqrt(dx * dx + dy * dy);
            double Etx = (E_elec + E_amp * d * d) * PACKET_SIZE;

            int r = d_relaysCount ? d_relaysCount[g] : 0;
            e_inter = (double)r * E_rx + (double)(r + 1) * Etx;
        }
        else {
            // invalid next hop, preserve behavior: skip
            continue;
        }

        double total = e_intra + e_inter;
        if (!(total > 0.0)) continue;      // excludes zero/NaN/neg
        if (!(energy > 0.0)) continue;     // gateway without energy is ignored

        double lifetime = energy / total;
        if (!isfinite(lifetime) || lifetime < 0.0) lifetime = 0.0;

        anyValid = true;
        if (lifetime < minLifetime) minLifetime = lifetime;
    }

    d_fitness[pid] = anyValid ? minLifetime : 0.0;
}

// --------------------------- update personal best (per particle) ---------------------------
__global__ void updatePersonalBestKernel_Block(
    const double* d_positions,
    double* d_pbest,
    const double* d_fitness,
    double* d_pbestFitness,
    int totalParticles,
    int numSensors)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= totalParticles) return;

    double f = d_fitness[pid];
    if (f > d_pbestFitness[pid]) {
        d_pbestFitness[pid] = f;
        size_t base = (size_t)pid * numSensors;
        for (int s = 0; s < numSensors; ++s)
            d_pbest[base + s] = d_positions[base + s];
    }
}

// --------------------------- per-block reduction to find block-local gbest ---------------------------
__global__ void computeBlockGBestKernel(
    const double* d_pbestFitness, // totalParticles
    const double* d_pbest,        // totalParticles * numSensors
    double* d_gbest_blocks,       // numBlocks * numSensors
    double* d_blockBestFitness,   // numBlocks
    int swarmPerBlock,
    int numSensors)
{
    extern __shared__ double sdata[]; // shared for fitness
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int baseParticle = block * swarmPerBlock;

    // load into shared
    double myFit = -1e300;
    if (tid < swarmPerBlock) myFit = d_pbestFitness[baseParticle + tid];
    sdata[tid] = myFit;
    __syncthreads();

    // reduction within block (swarmPerBlock assumed <= blockDim.x and power of two preferred)
    int stride = blockDim.x / 2;
    for (; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < swarmPerBlock) {
            if (sdata[tid + stride] > sdata[tid]) sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double bestFit = sdata[0];
        d_blockBestFitness[block] = bestFit;

        // find index of that best particle (linear scan within block)
        int bestIdx = -1;
        for (int i = 0; i < swarmPerBlock; ++i) {
            if (d_pbestFitness[baseParticle + i] == bestFit) { bestIdx = baseParticle + i; break; }
        }
        if (bestIdx >= 0) {
            // copy pbest -> block gbest
            size_t bytes = sizeof(double) * (size_t)numSensors;
            const double* src = d_pbest + (size_t)bestIdx * numSensors;
            double* dst = d_gbest_blocks + (size_t)block * numSensors;
            for (int s = 0; s < numSensors; ++s) dst[s] = src[s];
        }
    }
}

// --------------------------- update particles kernel (per particle) uses block-local gbest ---------------------------
__global__ void updateParticlesKernel_Block(
    double* d_positions,
    double* d_velocities,
    const double* d_pbest,
    const double* d_gbest_blocks,
    curandState* d_randStates,
    int totalParticles,
    int numSensors,
    double w, double c1, double c2)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x; // threadIdx.x corresponds to particle index within block
    if (pid >= totalParticles) return;

    int block = blockIdx.x; // block-local gbest index
    size_t base = (size_t)pid * numSensors;
    const double* gbest = d_gbest_blocks + (size_t)block * numSensors;

    curandState local = d_randStates[pid];

    for (int s = 0; s < numSensors; ++s) {
        int idx = base + s;
        double r1 = curand_uniform_double(&local);
        double r2 = curand_uniform_double(&local);

        double vel = w * d_velocities[idx] + c1 * r1 * (d_pbest[idx] - d_positions[idx]) + c2 * r2 * (gbest[s] - d_positions[idx]);
        if (vel > 0.5) vel = 0.5;
        if (vel < -0.5) vel = -0.5;
        d_velocities[idx] = vel;

        double pos = d_positions[idx] + vel;
        if (pos < 0.0) pos = 0.0;
        if (pos > 1.0) pos = 1.0;
        d_positions[idx] = pos;
    }

    d_randStates[pid] = local;
}

// --------------------------- Class implementation ---------------------------

ClusteringPSO_CUDA_Block::ClusteringPSO_CUDA_Block(Network& net_,
    const std::vector<int>& nextHopHost_,
    const std::vector<double>& clusterRadiiHost_,
    int swarmPerBlock_, int numBlocks_, int iterations_)
    : net(net_), nextHopHost(nextHopHost_), clusterRadiiHost(clusterRadiiHost_),
    swarmPerBlock(swarmPerBlock_), numBlocks(numBlocks_), iterations(iterations_)
{
    numSensors = net.numSensors;
    numGateways = net.numGateways;
    totalParticles = swarmPerBlock * numBlocks;
}

ClusteringPSO_CUDA_Block::~ClusteringPSO_CUDA_Block() {
    freeMemory();
}

void ClusteringPSO_CUDA_Block::allocateMemory() {
    size_t particles = (size_t)totalParticles;
    size_t sensors = (size_t)numSensors;

    CUDA_CALL_BLOCK(cudaMalloc(&d_positions, sizeof(double) * particles * sensors));
    CUDA_CALL_BLOCK(cudaMalloc(&d_velocities, sizeof(double) * particles * sensors));
    CUDA_CALL_BLOCK(cudaMalloc(&d_pbest, sizeof(double) * particles * sensors));
    CUDA_CALL_BLOCK(cudaMalloc(&d_pbestFitness, sizeof(double) * particles));
    CUDA_CALL_BLOCK(cudaMalloc(&d_gbest_blocks, sizeof(double) * (size_t)numBlocks * sensors));
    CUDA_CALL_BLOCK(cudaMalloc(&d_blockBestFitness, sizeof(double) * numBlocks));
    CUDA_CALL_BLOCK(cudaMalloc(&d_fitness, sizeof(double) * particles)); // d_fitness sized as totalParticles
    CUDA_CALL_BLOCK(cudaMalloc(&d_randStates, sizeof(curandState) * particles));

    // network arrays
    CUDA_CALL_BLOCK(cudaMalloc(&d_nodes, sizeof(NodeGPU) * (numGateways + numSensors)));
    CUDA_CALL_BLOCK(cudaMalloc(&d_nextHop, sizeof(int) * numGateways));
    CUDA_CALL_BLOCK(cudaMalloc(&d_clusterRadii, sizeof(double) * numGateways));
}

void ClusteringPSO_CUDA_Block::freeMemory() {

    auto safeFree = [&](auto &ptr) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::cerr << "[CUDA][ERROR] cudaFree FAILED: "
                          << cudaGetErrorString(err) << "\n";
            }
            ptr = nullptr;
        }
    };

    safeFree(d_positions);
    safeFree(d_velocities);
    safeFree(d_pbest);
    safeFree(d_pbestFitness);
    safeFree(d_gbest_blocks);
    safeFree(d_blockBestFitness);
    safeFree(d_randStates);
    safeFree(d_nodes);
    safeFree(d_nextHop);
    safeFree(d_clusterRadii);
    safeFree(d_fitness);
	safeFree(d_relaysCount);
	safeFree(d_sensorOffsets);
	safeFree(d_sensorAdj);
}

void ClusteringPSO_CUDA_Block::copyNetworkToDevice() {
    std::vector<NodeGPU> hostNodes;
    hostNodes.reserve(numGateways + numSensors);
    for (int i = 0; i < numGateways + numSensors; ++i) {
        const Node& n = net.nodes[i];
        NodeGPU ng(n.x, n.y, n.energy, i, n.isGateway ? 1 : 0);
        hostNodes.push_back(ng);
    }
    CUDA_CALL_BLOCK(cudaMemcpy(d_nodes, hostNodes.data(), sizeof(NodeGPU) * hostNodes.size(), cudaMemcpyHostToDevice));
    std::vector<int> h_next = nextHopHost;
    CUDA_CALL_BLOCK(cudaMemcpy(d_nextHop, h_next.data(), sizeof(int) * numGateways, cudaMemcpyHostToDevice));
    std::vector<double> h_radii = clusterRadiiHost;
    CUDA_CALL_BLOCK(cudaMemcpy(d_clusterRadii, h_radii.data(), sizeof(double) * numGateways, cudaMemcpyHostToDevice));
}

void ClusteringPSO_CUDA_Block::initializeParticles(unsigned long seed) {
    // init pbestFitness to very small
    {
        std::vector<double> h_init((size_t)totalParticles, -1e300);
        CUDA_CALL_BLOCK(cudaMemcpy(d_pbestFitness, h_init.data(), sizeof(double) * totalParticles, cudaMemcpyHostToDevice));
    }

    // kernel init
    size_t totalElems = (size_t)totalParticles * (size_t)numSensors;
    int t = 256;
    int b = (int)((totalElems + t - 1) / t);
    initParticlesBlockKernel << <b, t >> > (d_positions, d_velocities, totalParticles, numSensors, seed);
    CUDA_CALL_BLOCK(cudaGetLastError());
    CUDA_CALL_BLOCK(cudaDeviceSynchronize());

    // copy positions -> pbest
    CUDA_CALL_BLOCK(cudaMemcpy(d_pbest, d_positions, sizeof(double) * totalParticles * numSensors, cudaMemcpyDeviceToDevice));

    // init curand states for each particle (one state per particle)
    int threads = 256;
    int blocks = (totalParticles + threads - 1) / threads;
    initCurandKernel << <blocks, threads >> > (d_randStates, seed, totalParticles);
    CUDA_CALL_BLOCK(cudaGetLastError());
    CUDA_CALL_BLOCK(cudaDeviceSynchronize());
}

void ClusteringPSO_CUDA_Block::mergeBlockGBestToHostAndSelect() {
    // copies d_blockBestFitness and d_gbest_blocks to host and choose best across blocks
    std::vector<double> h_blockFitness(numBlocks);
    CUDA_CALL_BLOCK(cudaMemcpy(h_blockFitness.data(), d_blockBestFitness, sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));
    std::vector<double> h_blockGbest((size_t)numBlocks * numSensors);
    CUDA_CALL_BLOCK(cudaMemcpy(h_blockGbest.data(), d_gbest_blocks, sizeof(double) * numBlocks * numSensors, cudaMemcpyDeviceToHost));

    // choose best block
    double bestVal = -1e300;
    bestBlockIdx = -1;

    for (int b = 0; b < numBlocks; ++b) {
        if (h_blockFitness[b] > bestVal) { bestVal = h_blockFitness[b]; bestBlockIdx = b; }
    }

    if (bestBlockIdx >= 0) {
        h_gbest_cache.resize(numSensors);
        for (int s = 0; s < numSensors; ++s)
            h_gbest_cache[s] = h_blockGbest[(size_t)bestBlockIdx * numSensors + s];
    }
    else {
        // fallback: zeros
        h_gbest_cache.assign(numSensors, 0.5);
    }
}

void ClusteringPSO_CUDA_Block::run() {
    std::cout << "[CUDA][ClusteringPSO_Block] Iniciando (blocks=" << numBlocks
        << ", swarmPerBlock=" << swarmPerBlock << ")...\n";

    // ================================================================
    // CONFIGURAÇÃO DO LOOP (ESCOLHA AQUI)
    // ================================================================

    #define USE_PSO_TARGET_CRITERIA_BLOCK     // loop novo com alvo (while true)
    //#define USE_PSO_FIXED_ITER_BLOCK        // loop antigo (for fixo)

    #if !defined(USE_PSO_TARGET_CRITERIA_BLOCK) && !defined(USE_PSO_FIXED_ITER_BLOCK)
        #define USE_PSO_FIXED_ITER_BLOCK
    #endif

// ====================================================================
// INICIALIZAÇÃO (SEU CÓDIGO ORIGINAL – NADA ALTERADO)
// ====================================================================

    std::vector<double> host_clusterRadii(numGateways);
    if (!clusterRadiiHost.empty()) {
        for (int g = 0; g < numGateways; ++g)
            host_clusterRadii[g] = clusterRadiiHost[g];
    }
    else {
        CUDA_CALL_BLOCK(cudaMemcpy(host_clusterRadii.data(),
            d_clusterRadii,
            sizeof(double) * (size_t)numGateways,
            cudaMemcpyDeviceToHost));
    }

    // sensor adjacency (host)
    std::vector<int> sensorOffsets(numSensors + 1);
    std::vector<int> sensorAdj;
    sensorOffsets[0] = 0;
    sensorAdj.reserve((size_t)numSensors * 8);

    for (int s = 0; s < numSensors; ++s) {
        int count = 0;
        const Node& sensor = net.nodes[numGateways + s];

        for (int g = 0; g < numGateways; ++g) {
            const Node& gw = net.nodes[g];
            double dx = gw.x - sensor.x;
            double dy = gw.y - sensor.y;
            double d2 = dx * dx + dy * dy;
            double r = host_clusterRadii[g];

            if (d2 <= r * r) {
                sensorAdj.push_back(g);
                count++;
            }
        }

        sensorOffsets[s + 1] = sensorOffsets[s] + count;
    }

    allocateMemory();
    copyNetworkToDevice();
    initializeParticles();

    CUDA_CALL_BLOCK(cudaMalloc(&d_sensorOffsets, sizeof(int) * (numSensors + 1)));
    CUDA_CALL_BLOCK(cudaMalloc(&d_sensorAdj, sizeof(int) * (sensorAdj.size())));
    CUDA_CALL_BLOCK(cudaMemcpy(d_sensorOffsets, sensorOffsets.data(),
        sizeof(int) * (numSensors + 1), cudaMemcpyHostToDevice));
    CUDA_CALL_BLOCK(cudaMemcpy(d_sensorAdj, sensorAdj.data(),
        sizeof(int) * sensorAdj.size(), cudaMemcpyHostToDevice));
    sensorAdjAllocated = true;

    dim3 grid(numBlocks);
    dim3 block(swarmPerBlock);

    h_bestHistory.clear();
    h_bestHistory.reserve(iterations);

    // relaysCount
    std::vector<int> h_relays(numGateways, 0);
    for (int src = 0; src < numGateways; ++src) {
        int nh = nextHopHost[src];
        if (nh >= 0 && nh < numGateways) h_relays[nh]++;
    }

    CUDA_CALL_BLOCK(cudaMalloc(&d_relaysCount, sizeof(int) * numGateways));
    CUDA_CALL_BLOCK(cudaMemcpy(d_relaysCount, h_relays.data(),
        sizeof(int) * numGateways, cudaMemcpyHostToDevice));

    // parâmetros usados apenas na versão target
    const bool   stopOnTarget = true;
    const double targetFitness = 2000.0;
    const int    maxIter = 100000;
    const int    maxAllowedIterations = 200000;
    const int    stagnationLimit = 2000;
    const double maxWallTimeMs = 1000.0 * 60.0 * 30.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    int lastImprovementIter = 0;
    double bestSoFar = -1e300;

    // ====================================================================
    // LOOP PRINCIPAL – ESCOLHIDO VIA #define ACIMA
    // ====================================================================


    // ===============================
    //     LOOP NOVO — TARGET (WHILE)
    // ===============================
#if defined(USE_PSO_TARGET_CRITERIA_BLOCK)

    int it = 0;
    while (true)
    {
        // ================================
        // (A) EVALUATE
        // ================================
        size_t sharedMemBytes = sizeof(double) * (size_t)numGateways * 4;

        evaluateClustersBlockKernel << <numBlocks, swarmPerBlock, sharedMemBytes >> > (
            d_positions,
            d_fitness,
            totalParticles,
            numSensors,
            numGateways,
            d_nodes,
            d_nextHop,
            d_relaysCount,
            d_clusterRadii,
            d_sensorOffsets,
            d_sensorAdj,
            net.bs.x,
            net.bs.y
            );
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        // ================================
        // (B) PERSONAL BEST
        // ================================
        updatePersonalBestKernel_Block << <numBlocks, swarmPerBlock >> > (
            d_positions, d_pbest, d_fitness, d_pbestFitness,
            totalParticles, numSensors);
        CUDA_CALL_BLOCK(cudaGetLastError());

        // ================================
        // (C) LOCAL GBEST POR BLOCO
        // ================================
        size_t sharedBytes2 = sizeof(double) * (size_t)swarmPerBlock;
        computeBlockGBestKernel << <numBlocks, swarmPerBlock, sharedBytes2 >> > (
            d_pbestFitness, d_pbest, d_gbest_blocks, d_blockBestFitness,
            swarmPerBlock, numSensors);
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        // merge block gbests → host
        mergeBlockGBestToHostAndSelect();

        // ================================
        // (D) UPDATE PARTICLES
        // ================================
        updateParticlesKernel_Block << <numBlocks, swarmPerBlock >> > (
            d_positions, d_velocities, d_pbest, d_gbest_blocks, d_randStates,
            totalParticles, numSensors,
            0.7968, 1.4962, 1.4962
            );
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        // ================================
        // (E) GET GLOBAL BEST
        // ================================
        double mergedBestVal = -1e300;
        {
            std::vector<double> h_blockFitness(numBlocks);
            CUDA_CALL_BLOCK(cudaMemcpy(h_blockFitness.data(), d_blockBestFitness,
                sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));

            for (int b = 0; b < numBlocks; ++b)
                if (h_blockFitness[b] > mergedBestVal)
                    mergedBestVal = h_blockFitness[b];
        }

        if (gbestTimeline.empty() || mergedBestVal > gbestTimeline.back().second)
        {
            auto tNow = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();

            gbestTimeline.emplace_back(ms, mergedBestVal);
            lastImprovementIter = it;
            bestSoFar = mergedBestVal;
        }

        h_bestHistory.push_back(mergedBestVal);

        // ================================
        // (F) CRITÉRIOS DE PARADA
        // ================================
        if (stopOnTarget && mergedBestVal >= targetFitness) {
            auto tNow = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
            printf("Parou: atingiu targetFitness em %.2f ms (iter=%d)\n", ms, it);
            break;
        }

        if (it >= maxIter) {
            printf("Parou: maxIter atingido.\n");
            break;
        }

        if ((it - lastImprovementIter) >= stagnationLimit) {
            printf("Parou: estagnação. Fitness: %.2f\n", mergedBestVal);
            break;
        }

        if (it >= maxAllowedIterations) {
            printf("Parou: maxAllowedIterations.\n");
            break;
        }

        {
            auto tNow = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(tNow - t0).count();
            if (elapsed >= maxWallTimeMs) {
                printf("Parou: timeout.\n");
                break;
            }
        }

        ++it;
    }

    // ===============================
    //     LOOP ANTIGO — FIXO (FOR)
    // ===============================
#elif defined(USE_PSO_FIXED_ITER_BLOCK)

    for (int it = 0; it < iterations; ++it)
    {
        size_t sharedMemBytes = sizeof(double) * (size_t)numGateways * 4;

        evaluateClustersBlockKernel << <numBlocks, swarmPerBlock, sharedMemBytes >> > (
            d_positions,
            d_fitness,
            totalParticles,
            numSensors,
            numGateways,
            d_nodes,
            d_nextHop,
            d_relaysCount,
            d_clusterRadii,
            d_sensorOffsets,
            d_sensorAdj,
            net.bs.x,
            net.bs.y
            );
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        updatePersonalBestKernel_Block << <numBlocks, swarmPerBlock >> > (
            d_positions, d_pbest, d_fitness, d_pbestFitness,
            totalParticles, numSensors);
        CUDA_CALL_BLOCK(cudaGetLastError());

        size_t sharedBytes2 = sizeof(double) * (size_t)swarmPerBlock;
        computeBlockGBestKernel << <numBlocks, swarmPerBlock, sharedBytes2 >> > (
            d_pbestFitness, d_pbest, d_gbest_blocks, d_blockBestFitness,
            swarmPerBlock, numSensors);
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        mergeBlockGBestToHostAndSelect();

        updateParticlesKernel_Block << <numBlocks, swarmPerBlock >> > (
            d_positions, d_velocities, d_pbest, d_gbest_blocks,
            d_randStates, totalParticles, numSensors,
            0.7968, 1.4962, 1.4962
            );
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        double mergedBestVal = -1e300;
        {
            std::vector<double> h_blockFitness(numBlocks);
            CUDA_CALL_BLOCK(cudaMemcpy(h_blockFitness.data(), d_blockBestFitness,
                sizeof(double) * numBlocks,
                cudaMemcpyDeviceToHost));

            for (int b = 0; b < numBlocks; ++b)
                if (h_blockFitness[b] > mergedBestVal)
                    mergedBestVal = h_blockFitness[b];
        }

        if (gbestTimeline.empty() || mergedBestVal > gbestTimeline.back().second) {
            auto tNow = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
            gbestTimeline.emplace_back(ms, mergedBestVal);
        }

        h_bestHistory.push_back(mergedBestVal);
    }

#endif  // escolha do loop


    // ====================================================================
    // FINALIZAÇÃO — SEU CÓDIGO ORIGINAL
    // ====================================================================

    auto tEnd = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(tEnd - t0).count();

    mergeBlockGBestToHostAndSelect();

    // export converge
    {
        std::ofstream csv("pso_convergence_gpu_clustering_block.csv");
        csv << "iteration,best_fitness\n";
        for (int i = 0; i < (int)h_bestHistory.size(); ++i)
            csv << i << "," << h_bestHistory[i] << "\n";
    }

    // export timeline
    {
        std::ofstream out("gbest_timeline_gpu_block.csv");
        out << "time_ms,fitness\n";
        for (auto& p : gbestTimeline)
            out << p.first << "," << p.second << "\n";
        out << "END," << total_ms << "\n";
    }

    std::vector<double> bestPos(numSensors);
    CUDA_CALL_BLOCK(cudaMemcpy(
        bestPos.data(),
        d_gbest_blocks + (size_t)bestBlockIdx * (size_t)numSensors,
        sizeof(double) * numSensors,
        cudaMemcpyDeviceToHost
    ));

    // decode clustering final
    std::vector<int> assignmentGPUBlock(numSensors);
    for (int s = 0; s < numSensors; ++s) {
        int start = sensorOffsets[s];
        int end = sensorOffsets[s + 1];
        int deg = end - start;

        int assigned = -1;
        if (deg > 0) {
            int pick = (int)(bestPos[s] * deg);
            if (pick >= deg) pick = deg - 1;
            assigned = sensorAdj[start + pick];
        }

        assignmentGPUBlock[s] = assigned;
    }

    exportNetworkAndLinksToCSV(
        net,
        "gpu_block_network.csv",
        nextHopHost,
        assignmentGPUBlock,
        clusterRadiiHost
    );

    if (sensorAdjAllocated) {
        CUDA_CALL_BLOCK(cudaFree(d_sensorOffsets));
        CUDA_CALL_BLOCK(cudaFree(d_sensorAdj));

        d_sensorOffsets = nullptr;   // <--- ESSENCIAL
        d_sensorAdj = nullptr;   // <--- ESSENCIAL
        sensorAdjAllocated = false;
    }


    freeMemory();
}



std::vector<double> ClusteringPSO_CUDA_Block::getGBestHost() const {
    return h_gbest_cache;
}

#endif // USE_CUDA
