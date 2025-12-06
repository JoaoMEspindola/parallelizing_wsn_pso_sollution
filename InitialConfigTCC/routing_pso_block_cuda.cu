#include "routing_pso_block_cuda.hpp"
#include "utils.hpp"
#include "energy.hpp"
#include "energy_gpu.hpp"
#include "compact_graph.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>

// small safety
#ifndef PACKET_SIZE
#define PACKET_SIZE 4000.0
#endif

// -------------------------------
// kernels (block-per-swarm style)
// -------------------------------

// Kernel A: evaluate fitness per particle (thread)
// Note: each thread maps to a *global* particle index = blockIdx.x * ppr + threadIdx.x
// This kernel is basically the same logic as your evaluateParticlesKernel (thread-per-particle),
// but is included here so the block-per-swarm flow is self-contained.

__global__ void finalReduceGlobalKernel(
    const double* d_blockBestFitness, // numBlocks
    const double* d_blockGbest,       // numBlocks * numGateways
    double* d_gbest,                  // numGateways (out)
    int numBlocks,
    int numGateways)
{
    // single-block kernel expected; we launch with <<<1, min(1024, numBlocks)>>> or <<<1, 256>>>
    extern __shared__ double sfit[]; // size >= blockDim.x
    int tid = threadIdx.x;
    int lane = tid;

    // load fitness into shared (pad rest with -inf)
    double val = (lane < numBlocks) ? d_blockBestFitness[lane] : -1e300;
    sfit[lane] = val;
    __syncthreads();

    // parallel reduction to find max fitness and index
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            if (sfit[lane + offset] > sfit[lane]) sfit[lane] = sfit[lane + offset];
        }
        __syncthreads();
    }

    // thread 0 finds the index of best (linear scan small N)
    if (lane == 0) {
        double bestVal = sfit[0];
        int bestIdx = 0;
        for (int i = 0; i < numBlocks; ++i) {
            if (d_blockBestFitness[i] == bestVal) { bestIdx = i; break; }
        }
        // copy the chosen block's pbest into d_gbest
        const double* src = d_blockGbest + (size_t)bestIdx * (size_t)numGateways;
        for (int j = 0; j < numGateways; ++j) d_gbest[j] = src[j];
    }
}

__global__ void evaluateParticlesBlockKernel(
    const double* d_positions,  // swarmSize * numGateways
    double* d_fitness,          // swarmSize
    int swarmSize,
    int numGateways,
    const NodeGPU* d_nodes,     // numGateways + sensors if needed
    const int* d_offsets,       // compact graph offsets (optional)
    const int* d_adjacency,     // compact graph adjacency (optional)
    double bsx, double bsy,
    double gatewayRange,
    int particlesPerRevoada)
{
    int blockId = blockIdx.x;
    int localTid = threadIdx.x;
    int gid = blockId * particlesPerRevoada + localTid; // global particle id
    if (gid >= swarmSize) return;

    const double* pos = d_positions + (size_t)gid * (size_t)numGateways;

    // local small arrays on stack (numGateways expected small: ex 60)
    // nextHop and recvCount are per thread
    extern __shared__ int smem_int[]; // not used here; kept for signature compatibility
    // stack arrays (safe if numGateways small — typical in your sim)
    // fallback to conservative: limit to 1024 - but using dynamic memory in device is bad.
    // we assume numGateways <= 1024 (practical).
    int nextHopStack[128]; // adjust if you expect >128 gateways; else adapt to shared mem
    int recvCountStack[128];
    int ng = numGateways;
    if (ng > 128) {
        // fallback safe behavior: mark fitness zero if too large
        if (gid == 0) printf("[evaluateParticlesBlockKernel] numGateways (%d) > 128, increase stack or refactor\n", ng);
        d_fitness[gid] = 0.0;
        return;
    }
    for (int i = 0; i < ng; ++i) { nextHopStack[i] = -1; recvCountStack[i] = 0; }

    // decode nextHop using adjacency (if adjacency provided) OR simple mapping (pos->index)
    for (int g = 0; g < ng; ++g) {
        double v = pos[g];
        int selected = -1;
        if (d_offsets != nullptr && d_adjacency != nullptr) {
            int start = d_offsets[g];
            int end = d_offsets[g + 1];
            int deg = end - start;
            if (deg > 0) {
                int idx = min((int)(v * deg), deg - 1);
                selected = d_adjacency[start + idx];
                if (selected < 0 || selected >= ng) selected = -1;
            }
        }
        else {
            // fallback: continuous position -> map to gateway index or -1 (approx)
            if (v > 0.95) selected = -1; // high values interpret as BS
            else {
                int idxEstimate = min((int)(v * ng), ng - 1);
                selected = idxEstimate;
            }
        }
        nextHopStack[g] = selected;
    }

    // recv counts
    for (int s = 0; s < ng; ++s) {
        int nh = nextHopStack[s];
        if (nh >= 0 && nh < ng) ++recvCountStack[nh];
    }

    // compute minLifetime across gateways
    const double E_elec = 50e-9;
    const double E_amp = 10e-12;
    const double k = PACKET_SIZE;

    double minLifetime = 1e30;
    bool anyValid = false;
    for (int g = 0; g < ng; ++g) {
        const NodeGPU& gw = d_nodes[g];
        int nh = nextHopStack[g];

        double dist;
        if (nh == -1) {
            dist = distanceGPU(gw.x, gw.y, bsx, bsy);
        }
        else {
            const NodeGPU& next = d_nodes[nh];
            dist = distanceGPU(gw.x, gw.y, next.x, next.y);
        }

        if (nh != -1 && dist > gatewayRange) { minLifetime = 0.0; continue; }

        double E_tx = (E_elec + E_amp * dist * dist) * k;
        double E_rx = E_elec * k;
        int n_recv = recvCountStack[g];

        double E_total = (n_recv + 1) * E_tx + n_recv * E_rx;
        if (!(E_total > 0.0) || !isfinite(E_total)) continue;

        double lifetime = gw.energy / E_total;
        if (!isfinite(lifetime) || lifetime < 0.0) lifetime = 0.0;

        anyValid = true;
        if (lifetime < minLifetime) minLifetime = lifetime;
    }

    if (!anyValid || !isfinite(minLifetime) || minLifetime > 1e29) minLifetime = 0.0;
    d_fitness[gid] = minLifetime;
}

// Kernel B: update personal best (per particle) - same as before
__global__ void updatePersonalBestKernel_Block(
    const double* d_positions,
    double* d_pbest,
    const double* d_fitness,
    double* d_pbestFitness,
    int swarmSize,
    int numGateways,
    int particlesPerRevoada)
{
    int blockId = blockIdx.x;
    int localTid = threadIdx.x;
    int gid = blockId * particlesPerRevoada + localTid;
    if (gid >= swarmSize) return;

    double fit = d_fitness[gid];
    if (fit > d_pbestFitness[gid]) {
        d_pbestFitness[gid] = fit;
        size_t base = (size_t)gid * (size_t)numGateways;
        for (int j = 0; j < numGateways; ++j)
            d_pbest[base + j] = d_positions[base + j];
    }
}

// Kernel C: block-level reduction to select best particle inside each revoada (block)
// Writes d_blockBestFitness[blockIdx] and copies the corresponding pbest into d_blockGbest[blockIdx * numGateways + j]
__global__ void blockReduceBestKernel(
    const double* d_pbestFitness, // swarmSize
    const double* d_pbest,        // swarmSize * numGateways
    double* d_blockBestFitness,   // numRevoadas
    double* d_blockGbest,         // numRevoadas * numGateways
    int swarmSize,
    int numGateways,
    int particlesPerRevoada)
{
    extern __shared__ double sdata[]; // will hold fitness values (as double)
    int localTid = threadIdx.x;
    int blockId = blockIdx.x;
    int gid = blockId * particlesPerRevoada + localTid;
    double myFit = (gid < swarmSize) ? d_pbestFitness[gid] : -1e300;
    sdata[localTid] = myFit;
    __syncthreads();

    // reduction in shared mem (power-of-two assumed for particlesPerRevoada or safe loop)
    int stride = blockDim.x / 2;
    while (stride > 0) {
        if (localTid < stride) {
            if (sdata[localTid + stride] > sdata[localTid]) sdata[localTid] = sdata[localTid + stride];
        }
        __syncthreads();
        stride >>= 1;
    }

    if (localTid == 0) {
        double bestFit = sdata[0];
        // Find index of that bestFit among threads of this block (linear scan)
        int bestLocalIndex = -1;
        for (int t = 0; t < particlesPerRevoada; ++t) {
            int idCheck = blockId * particlesPerRevoada + t;
            if (idCheck >= swarmSize) continue;
            if (d_pbestFitness[idCheck] == bestFit) { bestLocalIndex = t; break; }
        }
        if (bestLocalIndex < 0) {
            // nothing valid
            d_blockBestFitness[blockId] = -1e300;
        }
        else {
            int bestGlobal = blockId * particlesPerRevoada + bestLocalIndex;
            d_blockBestFitness[blockId] = d_pbestFitness[bestGlobal];
            // copy pbest -> blockGbest
            const double* src = d_pbest + (size_t)bestGlobal * numGateways;
            double* dst = d_blockGbest + (size_t)blockId * (size_t)numGateways;
            for (int j = 0; j < numGateways; ++j) dst[j] = src[j];
        }
    }
}

// Kernel D: update particles (each thread updates its particle) similar to your updateParticlesKernel
__global__ void updateParticlesKernel_Block(
    double* d_positions,
    double* d_velocities,
    const double* d_pbest,
    const double* d_gbest,
    curandState* d_randStates,
    int swarmSize,
    int numGateways,
    int particlesPerRevoada,
    double w, double c1, double c2)
{
    int blockId = blockIdx.x;
    int localTid = threadIdx.x;
    int gid = blockId * particlesPerRevoada + localTid;
    if (gid >= swarmSize) return;

    curandState localState = d_randStates[gid];

    size_t base = (size_t)gid * (size_t)numGateways;
    for (int g = 0; g < numGateways; ++g) {
        int idx = base + g;
        double r1 = curand_uniform_double(&localState);
        double r2 = curand_uniform_double(&localState);
        double vel = d_velocities[idx];
        double pos = d_positions[idx];
        double pbest = d_pbest[idx];
        double gbest = d_gbest[g];

        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos);
        if (vel > 0.5) vel = 0.5;
        if (vel < -0.5) vel = -0.5;
        pos += vel;
        if (pos < 0.0) pos = 0.0;
        if (pos > 1.0) pos = 1.0;
        d_velocities[idx] = vel;
        d_positions[idx] = pos;
    }

    d_randStates[gid] = localState;
}

// -------------------------------------------------
// Class implementation
// -------------------------------------------------

RoutingPSO_Block_CUDA::RoutingPSO_Block_CUDA(Network& net_, int swarmSize_, int iterations_, int numRevoadas_)
    : net(net_), swarmSize(swarmSize_), iterations(iterations_), numRevoadas(numRevoadas_)
{
    numGateways = net.numGateways;
    // particles per revoada (threads per block)
    particlesPerRevoada = (swarmSize + numRevoadas - 1) / numRevoadas;
    // ensure particlesPerRevoada is reasonable (CUDA block limit)
    if (particlesPerRevoada > 1024) {
        std::cerr << "[RoutingPSO_Block_CUDA] particlesPerRevoada (" << particlesPerRevoada << ") > 1024\n";
        particlesPerRevoada = 1024;
    }
}

RoutingPSO_Block_CUDA::~RoutingPSO_Block_CUDA() {
    freeMemory();
}

inline int highestPowerOfTwoLE(int x) {
    int p = 1;
    while (p << 1 <= x) p <<= 1;
    return p;
}


void RoutingPSO_Block_CUDA::allocateMemory() {
    size_t totalParticles = (size_t)swarmSize * (size_t)numGateways;
    CUDA_CALL(cudaMalloc(&d_positions, sizeof(double) * totalParticles));
    CUDA_CALL(cudaMalloc(&d_velocities, sizeof(double) * totalParticles));
    CUDA_CALL(cudaMalloc(&d_pbest, sizeof(double) * totalParticles));
    CUDA_CALL(cudaMalloc(&d_pbestFitness, sizeof(double) * swarmSize));
    CUDA_CALL(cudaMalloc(&d_fitness, sizeof(double) * swarmSize));
    CUDA_CALL(cudaMalloc(&d_gbest, sizeof(double) * numGateways));
    CUDA_CALL(cudaMalloc(&d_randStates, sizeof(curandState) * (size_t)swarmSize));

    // per-block buffers
    CUDA_CALL(cudaMalloc(&d_blockGbest, sizeof(double) * (size_t)numRevoadas * (size_t)numGateways));
    CUDA_CALL(cudaMalloc(&d_blockBestFitness, sizeof(double) * (size_t)numRevoadas));

    h_blockBestBuffer.resize(numRevoadas);
}

void RoutingPSO_Block_CUDA::freeMemory() {
    if (d_positions) CUDA_CALL(cudaFree(d_positions));
    if (d_velocities) CUDA_CALL(cudaFree(d_velocities));
    if (d_pbest) CUDA_CALL(cudaFree(d_pbest));
    if (d_pbestFitness) CUDA_CALL(cudaFree(d_pbestFitness));
    if (d_fitness) CUDA_CALL(cudaFree(d_fitness));
    if (d_gbest) CUDA_CALL(cudaFree(d_gbest));
    if (d_randStates) CUDA_CALL(cudaFree(d_randStates));
    if (d_blockGbest) CUDA_CALL(cudaFree(d_blockGbest));
    if (d_blockBestFitness) CUDA_CALL(cudaFree(d_blockBestFitness));

    d_positions = d_velocities = d_pbest = nullptr;
    d_pbestFitness = d_fitness = d_gbest = nullptr;
    d_randStates = nullptr;
    d_blockGbest = nullptr;
    d_blockBestFitness = nullptr;
}

void RoutingPSO_Block_CUDA::initializeParticles() {
    // init pbestFitness very low
    {
        std::vector<double> h_init(swarmSize, -1e300);
        CUDA_CALL(cudaMemcpy(d_pbestFitness, h_init.data(), sizeof(double) * swarmSize, cudaMemcpyHostToDevice));
    }

    // init positions/velocities deterministically without curand (simple)
    dim3 t(256);
    dim3 b(((size_t)swarmSize * (size_t)numGateways + t.x - 1) / t.x);
    // reuse the simple init kernel from your routing_pso_cuda if present, else do device->device copy from host random
    // here we'll generate on host and copy (robust)
    std::vector<double> h_pos((size_t)swarmSize * (size_t)numGateways);
    std::vector<double> h_vel((size_t)swarmSize * (size_t)numGateways);
    for (size_t i = 0; i < h_pos.size(); ++i) {
        double r = randDouble(0.0, 1.0);
        h_pos[i] = r;
        h_vel[i] = randDouble(-0.5, 0.5);
    }
    CUDA_CALL(cudaMemcpy(d_positions, h_pos.data(), sizeof(double) * h_pos.size(), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_velocities, h_vel.data(), sizeof(double) * h_vel.size(), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pbest, d_positions, sizeof(double) * h_pos.size(), cudaMemcpyDeviceToDevice));

    // init curand states (one per particle)
    dim3 t2(256);
    dim3 b2(((size_t)swarmSize + t2.x - 1) / t2.x);
    initCurandKernel << <b2, t2 >> > (d_randStates, 424242UL, swarmSize);
    CUDA_CALL(cudaDeviceSynchronize());

    // initialize global gbest with mid-values
    std::vector<double> h_gbest(numGateways, 0.5);
    CUDA_CALL(cudaMemcpy(d_gbest, h_gbest.data(), sizeof(double) * numGateways, cudaMemcpyHostToDevice));
}

std::vector<double> RoutingPSO_Block_CUDA::getGBestHost() const {
    std::vector<double> h_gbest(numGateways, 0.0);
    if (d_gbest) {
        CUDA_CALL(cudaMemcpy(h_gbest.data(), d_gbest, sizeof(double) * numGateways, cudaMemcpyDeviceToHost));
    }
    return h_gbest;
}

std::vector<int> RoutingPSO_Block_CUDA::decodeRoutingGBestToNextHop() const {
    std::vector<double> h_gbest = getGBestHost();
    std::vector<int> nextHop(numGateways, -1);
    for (int g = 0; g < numGateways; ++g) {
        std::vector<int> candidates = net.getNextHopCandidates(g);
        if (candidates.empty()) continue;
        double posVal = h_gbest[g];
        int idx = (int)(posVal * candidates.size());
        if (idx >= (int)candidates.size()) idx = (int)candidates.size() - 1;
        nextHop[g] = candidates[idx];
    }
    return nextHop;
}

void RoutingPSO_Block_CUDA::run() {
    // build compact graph if needed (copied from your routing version)
    CompactGraphHost hostGraph = buildCompactGraph(net);
    graphDev.totalNodes = (int)hostGraph.h_nodes.size();
    graphDev.totalEdges = (int)hostGraph.h_adjacency.size();
    CUDA_CALL(cudaMalloc(&graphDev.d_nodes, sizeof(NodeGPU) * graphDev.totalNodes));
    CUDA_CALL(cudaMalloc(&graphDev.d_offsets, sizeof(int) * hostGraph.h_offsets.size()));
    CUDA_CALL(cudaMalloc(&graphDev.d_adjacency, sizeof(int) * graphDev.totalEdges));
    CUDA_CALL(cudaMemcpy(graphDev.d_nodes, hostGraph.h_nodes.data(), sizeof(NodeGPU) * graphDev.totalNodes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(graphDev.d_offsets, hostGraph.h_offsets.data(), sizeof(int) * hostGraph.h_offsets.size(), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(graphDev.d_adjacency, hostGraph.h_adjacency.data(), sizeof(int) * graphDev.totalEdges, cudaMemcpyHostToDevice));
    graphAllocated = true;

    // allocate, init
    allocateMemory();
    initializeParticles();

    dim3 threads(particlesPerRevoada);
    dim3 blocks(numRevoadas);

    std::vector<double> h_bestHistory;
    h_bestHistory.reserve(iterations);

    for (int it = 0; it < iterations; ++it) {
        // (A) eval per-particle
        CUDA_CALL(cudaMemset(d_fitness, 0, sizeof(double) * swarmSize));
        evaluateParticlesBlockKernel << <blocks, threads >> > (
            d_positions, d_fitness, swarmSize, numGateways,
            graphDev.d_nodes, graphDev.d_offsets, graphDev.d_adjacency,
            net.bs.x, net.bs.y, net.gatewayRange, particlesPerRevoada);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // (B) update personal bests
        updatePersonalBestKernel_Block << <blocks, threads >> > (
            d_positions, d_pbest, d_fitness, d_pbestFitness, swarmSize, numGateways, particlesPerRevoada);
        CUDA_CALL(cudaGetLastError());

        // (C) reduce within each block to generate block bests (in GPU)
        size_t smemBytes = sizeof(double) * particlesPerRevoada;
        blockReduceBestKernel << <blocks, threads, smemBytes >> > (
            d_pbestFitness, d_pbest, d_blockBestFitness, d_blockGbest,
            swarmSize, numGateways, particlesPerRevoada);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // numa única chamada na GPU escolha global best entre blocks (numRevoadas)
        int fbThreads = 256;
        if (numRevoadas < fbThreads)
            fbThreads = highestPowerOfTwoLE(numRevoadas);

        finalReduceGlobalKernel << <1, fbThreads, sizeof(double)* fbThreads >> > (
            d_blockBestFitness, d_blockGbest, d_gbest, numRevoadas, numGateways);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());


        // (D) copy block bests to host, choose global best among blocks (host-level)
        CUDA_CALL(cudaMemcpy(h_blockBestBuffer.data(), d_blockBestFitness, sizeof(double) * numRevoadas, cudaMemcpyDeviceToHost));
        int bestBlockIdx = 0;
        double bestBlockFit = h_blockBestBuffer[0];
        for (int b = 1; b < numRevoadas; ++b) {
            if (h_blockBestBuffer[b] > bestBlockFit) { bestBlockFit = h_blockBestBuffer[b]; bestBlockIdx = b; }
        }
        // copy block's gbest (device->device) into global d_gbest
        CUDA_CALL(cudaMemcpy(d_gbest, d_blockGbest + (size_t)bestBlockIdx * numGateways, sizeof(double) * numGateways, cudaMemcpyDeviceToDevice));

        // (E) update particles using global gbest (per-block kernel)
        updateParticlesKernel_Block << <blocks, threads >> > (
            d_positions, d_velocities, d_pbest, d_gbest, d_randStates,
            swarmSize, numGateways, particlesPerRevoada, 0.7968, 1.4962, 1.4962);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        h_bestHistory.push_back(bestBlockFit);
    }

    // export convergence
    std::ofstream csv("pso_convergence_gpu_block.csv");
    csv << "iteration,best_fitness\n";
    for (int i = 0; i < (int)h_bestHistory.size(); ++i) csv << i << "," << h_bestHistory[i] << "\n";
    csv.close();

    // finalize
    if (graphAllocated) {
        CUDA_CALL(cudaFree(graphDev.d_nodes));
        CUDA_CALL(cudaFree(graphDev.d_offsets));
        CUDA_CALL(cudaFree(graphDev.d_adjacency));
        graphAllocated = false;
    }
    freeMemory();
}
