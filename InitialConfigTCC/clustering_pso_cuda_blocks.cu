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
    const int* __restrict__ d_sensorOffsets, // new
    const int* __restrict__ d_sensorAdj,     // new
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
    // ------------------------
    extern __shared__ double s_buf[]; // request 4 * numGateways * sizeof(double)
    double* s_gw_x = s_buf;
    double* s_gw_y = s_gw_x + numGateways;
    double* s_gw_E = s_gw_y + numGateways;
    double* s_r2 = s_gw_E + numGateways;

    // cooperative load
    for (int g = threadIdx.x; g < numGateways; g += blockDim.x) {
        const NodeGPU& gw = d_nodes[g];
        s_gw_x[g] = gw.x;
        s_gw_y[g] = gw.y;
        s_gw_E[g] = gw.energy;
        double r = d_clusterRadii ? d_clusterRadii[g] : 0.0;
        s_r2[g] = r * r;
    }
    __syncthreads();

    // clusterSizes per-thread (small footprint)
    // assume numSensors <= 65535; senão use uint32_t
    uint16_t clusterSizes[GW_LIMIT];
#pragma unroll 4
    for (int g = 0; g < numGateways; ++g) clusterSizes[g] = 0;

    // For each sensor, iterate only over its candidate gateways (from adjacency)
    for (int s = 0; s < numSensors; ++s) {
        double val = pos[s];

        int start = d_sensorOffsets[s];
        int end = d_sensorOffsets[s + 1];
        int deg = end - start;
        if (deg == 0) continue;

        // pick in [0,deg-1]
        int pick = (int)(val * (double)deg);
        if (pick >= deg) pick = deg - 1;

        int chosenGateway = d_sensorAdj[start + pick];
        if (chosenGateway >= 0 && chosenGateway < numGateways)
            clusterSizes[chosenGateway]++;
    }

    // -------------------------------------------------------------
    // Compute lifetime using clusterSizes (per-thread), gateways in shared
    // -------------------------------------------------------------
    const double E_elec = 50e-9;
    const double E_amp = 10e-12;
    const double E_rx = E_elec * PACKET_SIZE;
    const double E_agg = 5e-9 * PACKET_SIZE;

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
            double dx = gwx - bsx;
            double dy = gwy - bsy;
            double d2 = dx * dx + dy * dy;
            double Etx = (E_elec + E_amp * d2) * PACKET_SIZE;
            e_inter = Etx;
        }
        else if (nh >= 0 && nh < numGateways) {
            double dx = gwx - s_gw_x[nh];
            double dy = gwy - s_gw_y[nh];
            double d2 = dx * dx + dy * dy;
            double Etx = (E_elec + E_amp * d2) * PACKET_SIZE;

            int r = d_relaysCount[g];
            e_inter = (double)r * E_rx + (double)(r + 1) * Etx;
        }
        else {
            continue;
        }

        double total = e_intra + e_inter;
        if (total <= 0.0) continue;
        if (!(energy > 0.0)) continue;

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

    if (d_sensorOffsets) {
        CUDA_CALL_BLOCK(cudaFree(d_sensorOffsets));
        d_sensorOffsets = nullptr;
    }
    if (d_sensorAdj) {
        CUDA_CALL_BLOCK(cudaFree(d_sensorAdj));
        d_sensorAdj = nullptr;
    }
    sensorAdjAllocated = false;

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
    initCurandKernel << <blocks, threads >> > (d_randStates, seed ^ 0xABCDEFu, totalParticles);
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
    int bestBlock = -1;
    for (int b = 0; b < numBlocks; ++b) {
        if (h_blockFitness[b] > bestVal) { bestVal = h_blockFitness[b]; bestBlock = b; }
    }

    if (bestBlock >= 0) {
        h_gbest_cache.resize(numSensors);
        for (int s = 0; s < numSensors; ++s)
            h_gbest_cache[s] = h_blockGbest[(size_t)bestBlock * numSensors + s];
    }
    else {
        // fallback: zeros
        h_gbest_cache.assign(numSensors, 0.5);
    }
}

void ClusteringPSO_CUDA_Block::run() {
    std::cout << "[CUDA][ClusteringPSO_Block] Iniciando (blocks=" << numBlocks << ", swarmPerBlock=" << swarmPerBlock << ")...\n";

    // --- copiar cluster radii para host (se necessário) ---
    std::vector<double> host_clusterRadii(numGateways);
    // prefer host copy if available (clusterRadiiHost), senão fallback a device -> host copy
    if (!clusterRadiiHost.empty()) {
        for (int g = 0; g < numGateways; ++g) host_clusterRadii[g] = clusterRadiiHost[g];
    }
    else {
        CUDA_CALL_BLOCK(cudaMemcpy(host_clusterRadii.data(),
            d_clusterRadii,
            sizeof(double) * (size_t)numGateways,
            cudaMemcpyDeviceToHost));
    }


    // --- construir sensor -> gateway adjacency (HOST) ---
    std::vector<int> sensorOffsets(numSensors + 1);
    std::vector<int> sensorAdj;
    sensorOffsets[0] = 0;
    sensorAdj.reserve((size_t)numSensors * 8); // heurística

    // Assumo que você tem uma cópia host de nodes em net.h_nodes (gateway first, then sensors)
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


    // --- copiar adjacency para device ---
    CUDA_CALL_BLOCK(cudaMalloc(&d_sensorOffsets, sizeof(int) * (numSensors + 1)));
    CUDA_CALL_BLOCK(cudaMalloc(&d_sensorAdj, sizeof(int) * (sensorAdj.size())));
    CUDA_CALL_BLOCK(cudaMemcpy(d_sensorOffsets, sensorOffsets.data(), sizeof(int) * (numSensors + 1), cudaMemcpyHostToDevice));
    CUDA_CALL_BLOCK(cudaMemcpy(d_sensorAdj, sensorAdj.data(), sizeof(int) * (sensorAdj.size()), cudaMemcpyHostToDevice));
    sensorAdjAllocated = true;

    std::cout << "[CLUSTERING] sensorAdj built. totalAdj = " << sensorAdj.size() << "\n";

    // --- restante da inicialização (aloca e inicializa partículas, network->device etc) ---
    allocateMemory();
    copyNetworkToDevice();
    initializeParticles((unsigned long)time(nullptr));

    // threads por bloco (mantive seu valor inicial, ajuste se quiser)
    int threads = 32;

    dim3 grid(numBlocks);
    dim3 block(threads);

    h_bestHistory.clear();
    h_bestHistory.reserve(iterations);

    // pré-calcula relays (host) e copia
    std::vector<int> h_relays(numGateways, 0);
    for (int src = 0; src < numGateways; ++src) {
        int nh = nextHopHost[src]; // ou vetor que você tem
        if (nh >= 0 && nh < numGateways) h_relays[nh] += 1;
    }
    CUDA_CALL_BLOCK(cudaMalloc(&d_relaysCount, sizeof(int) * numGateways));
    CUDA_CALL_BLOCK(cudaMemcpy(d_relaysCount, h_relays.data(), sizeof(int) * numGateways, cudaMemcpyHostToDevice));

    // --- loop principal PSO ---
    for (int it = 0; it < iterations; ++it) {
        // evaluate (totalParticles threads = numBlocks * swarmPerBlock)
        int totalParticles_local = totalParticles;
        int tEval = threads;       // threads por bloco
        int bEval = numBlocks;     // cada bloco = uma revoada

        // --- shared memory bytes: apenas arrays dos gateways (x,y,E,r2) ---
        size_t sharedMemBytes = sizeof(double) * (size_t)numGateways * 4;

        // tempo do kernel (opcional)
        // --- Lançar kernel de avaliação que usa adjacency pré-computada ---
        evaluateClustersBlockKernel << <bEval, tEval, sharedMemBytes >> > (
            d_positions,
            d_fitness,
            totalParticles,
            numSensors,
            numGateways,
            d_nodes,          // device pointer com nodes (gateway + sensors) -> ajuste se usar graphDev.d_nodes
            d_nextHop,
            d_relaysCount,
            d_clusterRadii,
            d_sensorOffsets,  // novo
            d_sensorAdj,      // novo
            net.bs.x,
            net.bs.y
            );

        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        // --- update personal best (por partícula) ---
        updatePersonalBestKernel_Block << <numBlocks, threads >> > (d_positions, d_pbest, d_fitness, d_pbestFitness, totalParticles, numSensors);
        CUDA_CALL_BLOCK(cudaGetLastError());

        // --- compute block-local gbest (one block per revoada) ---
        size_t sharedBytes2 = (size_t)threads * sizeof(double);
        computeBlockGBestKernel << <numBlocks, threads, sharedBytes2 >> > (d_pbestFitness, d_pbest, d_gbest_blocks, d_blockBestFitness, swarmPerBlock, numSensors);
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        // --- merge block gbest to host and select best among blocks ---
        mergeBlockGBestToHostAndSelect();

        // --- update particles using block-local gbest ---
        updateParticlesKernel_Block << <numBlocks, threads >> > (d_positions, d_velocities, d_pbest, d_gbest_blocks, d_randStates, totalParticles, numSensors, 0.7968, 1.4962, 1.4962);
        CUDA_CALL_BLOCK(cudaGetLastError());
        CUDA_CALL_BLOCK(cudaDeviceSynchronize());

        // For logging: merged best
        double mergedBestVal = -1e300;
        {
            std::vector<double> h_blockFitness(numBlocks);
            CUDA_CALL_BLOCK(cudaMemcpy(h_blockFitness.data(), d_blockBestFitness, sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));
            for (int b = 0; b < numBlocks; ++b) if (h_blockFitness[b] > mergedBestVal) mergedBestVal = h_blockFitness[b];
        }
        h_bestHistory.push_back(mergedBestVal);

        if (it % 10 == 0) std::cout << "[CUDA][ClusteringPSO_Block] iter " << it << " merged_best = " << mergedBestVal << "\n";
    } // end iterations

    // final merge & cache
    mergeBlockGBestToHostAndSelect();

    // export convergence
    std::ofstream csv("pso_convergence_gpu_clustering_block.csv");
    csv << "iteration,best_fitness\n";
    for (int i = 0; i < (int)h_bestHistory.size(); ++i) csv << i << "," << h_bestHistory[i] << "\n";
    csv.close();

    std::cout << "[CUDA][ClusteringPSO_Block] finalizado. melhor fitness final = "
        << (h_bestHistory.empty() ? 0.0 : h_bestHistory.back()) << "\n";

    // --- liberar adjacency (se ainda alocado) ---
    if (sensorAdjAllocated) {
        CUDA_CALL_BLOCK(cudaFree(d_sensorOffsets));
        CUDA_CALL_BLOCK(cudaFree(d_sensorAdj));
        d_sensorOffsets = nullptr;
        d_sensorAdj = nullptr;
        sensorAdjAllocated = false;
    }

    freeMemory();
}


std::vector<double> ClusteringPSO_CUDA_Block::getGBestHost() const {
    return h_gbest_cache;
}

#endif // USE_CUDA
