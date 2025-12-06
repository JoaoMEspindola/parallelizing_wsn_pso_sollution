#ifdef USE_CUDA

#include "export.hpp"
#include "node_gpu.hpp"
#include "clustering_pso_cuda.hpp"
#include "cuda_common_kernels.hpp"

#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cmath>

#ifndef PACKET_SIZE
#define PACKET_SIZE 4000.0
#endif

// ----------------- kernels -----------------

// inicializa posições/velocidades aleatórias (cada thread gera para vários elementos)
__global__ void initParticlesClustKernel(double* d_positions, double* d_velocities,
    int swarmSize, int numSensors, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = swarmSize * numSensors;
    curandState state;
    // um gerador local simples (não ótimo, mas suficiente p/ init)
    int gid = idx;
    if (gid < total) {
        // simples LCG-ish based seed (determinístico por seed)
        unsigned long s = seed ^ (unsigned long)gid;
        // convert to pseudo-rand using xorshift64*
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        double r = (double)(s & 0xFFFF) / (double)0xFFFF;
        d_positions[gid] = r; // in [0,1)
        // velocity small random in [-0.5,0.5]
        d_velocities[gid] = r * 1.0 - 0.5;
    }
}

__global__ void assignSensorsKernel(
    const double* d_positions,   // [swarmSize * numSensors]
    int* d_assignment,           // [swarmSize * numSensors]
    int swarmSize,
    int numSensors,
    int numGateways,
    const NodeGPU* d_nodes,      // [numGateways + numSensors]
    const double* d_clusterRadii // [numGateways]
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)swarmSize * numSensors;
    if (idx >= total) return;

    int pid = idx / numSensors;
    int s = idx % numSensors;

    const NodeGPU& sensor = d_nodes[numGateways + s];
    double posVal = d_positions[pid * (size_t)numSensors + s];

    int chosen = -1;
    int candCount = 0;

    // encontra todos os gateways elegíveis
    for (int g = 0; g < numGateways; ++g) {
        double dx = sensor.x - d_nodes[g].x;
        double dy = sensor.y - d_nodes[g].y;
        double d = sqrt(dx * dx + dy * dy);
        if (d <= d_clusterRadii[g]) candCount++;
    }

    if (candCount == 0) {
        d_assignment[idx] = -1;
        return;
    }

    // seleciona gateway conforme posVal normalizado
    int pick = (int)(posVal * candCount);
    if (pick >= candCount) pick = candCount - 1;

    int current = 0;
    for (int g = 0; g < numGateways; ++g) {
        double dx = sensor.x - d_nodes[g].x;
        double dy = sensor.y - d_nodes[g].y;
        double d = sqrt(dx * dx + dy * dy);
        if (d <= d_clusterRadii[g]) {
            if (current == pick) {
                chosen = g;
                break;
            }
            current++;
        }
    }

    d_assignment[idx] = chosen;
}

__global__ void countClustersKernel(
    const int* d_assignment, // [swarmSize * numSensors]
    int* d_clusterSizes,     // [swarmSize * numGateways]
    int swarmSize,
    int numSensors,
    int numGateways)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)swarmSize * numSensors;
    if (idx >= total) return;

    int pid = idx / numSensors;
    int assigned = d_assignment[idx];

    if (assigned >= 0 && assigned < numGateways) {
        atomicAdd(&d_clusterSizes[(size_t)pid * numGateways + assigned], 1);
    }
}

__global__ void computeFitnessPerParticleKernel(
    const int* d_clusterSizes, // [swarmSize * numGateways]
    double* d_fitness,         // [swarmSize]
    int swarmSize,
    int numGateways,
    const NodeGPU* d_nodes,    // [numGateways + numSensors]
    const int* d_nextHop,      // [numGateways]
    const int* d_relaysCount,  // [numGateways]
    double bsx,
    double bsy,
    double gatewayRange)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= swarmSize) return;

    const int* cluster = d_clusterSizes + (size_t)pid * numGateways;

    const double E_elec = 50e-9;
    const double E_amp = 10e-12;
    const double E_rx = E_elec * PACKET_SIZE;
    const double E_agg = 5e-9 * PACKET_SIZE;

    double minLifetime = 1e300;
    bool valid = false;

    for (int g = 0; g < numGateways; ++g) {
        const NodeGPU& gw = d_nodes[g];

        double e_intra = cluster[g] * (E_rx + E_agg);
        double e_inter = 0.0;

        int nh = d_nextHop[g];
        if (nh == -1) {
            double dx = gw.x - bsx;
            double dy = gw.y - bsy;
            double d_bs = sqrt(dx * dx + dy * dy);
            e_inter = (E_elec + E_amp * d_bs * d_bs) * PACKET_SIZE;
        }
        else if (nh >= 0 && nh < numGateways) {
            const NodeGPU& nxt = d_nodes[nh];
            double dx = gw.x - nxt.x;
            double dy = gw.y - nxt.y;
            double dgw = sqrt(dx * dx + dy * dy);
            double E_tx = (E_elec + E_amp * dgw * dgw) * PACKET_SIZE;
            int relays = d_relaysCount[g];
            e_inter = relays * E_rx + (relays + 1) * E_tx;
        }
        else continue;

        double total = e_intra + e_inter;
        if (!(total > 0.0) || !isfinite(total)) continue;

        double lifetime = gw.energy / total;
        if (!isfinite(lifetime) || lifetime < 0.0) lifetime = 0.0;

        valid = true;
        if (lifetime < minLifetime) minLifetime = lifetime;
    }

    if (!valid || !isfinite(minLifetime) || minLifetime > 1e299)
        minLifetime = 0.0;

    d_fitness[pid] = minLifetime;
}


// evaluateClustersKernel (cada thread = 1 partícula)
// Mantive a versão local (cada thread usa arrays locais para clusterSizes)
__global__ void evaluateClustersKernel(
    const double* d_positions,     // size swarmSize * numSensors
    double* d_fitness,             // size swarmSize
    int swarmSize,
    int numSensors,
    int numGateways,
    const NodeGPU* d_nodes,        // size numGateways + numSensors
    const int* d_nextHop,          // size numGateways
    const double* d_clusterRadii,  // size numGateways
    double bsx,
    double bsy,
    double gatewayRange)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= swarmSize) return;

    // --- ARRAYS LOCAIS (sem new/delete, sem limite fixo global) ---
    extern __shared__ int sharedMem[];
    int* clusterSizes = sharedMem;
    int* relaysCount = clusterSizes + numGateways;
    int* candidates = relaysCount + numGateways;

    const double* pos = d_positions + (size_t)pid * numSensors;

    // Inicializações
    for (int g = 0; g < numGateways; ++g) {
        clusterSizes[g] = 0;
        relaysCount[g] = 0;
    }

    // === 1) Associação sensor → gateway ===
    for (int s = 0; s < numSensors; ++s) {
        double posVal = pos[s];
        int candCount = 0;
        const NodeGPU& sensor = d_nodes[numGateways + s];

        for (int g = 0; g < numGateways; ++g) {
            const NodeGPU& gw = d_nodes[g];
            double dx = sensor.x - gw.x;
            double dy = sensor.y - gw.y;
            double d = sqrt(dx * dx + dy * dy);
            if (d <= d_clusterRadii[g]) {
                candidates[candCount++] = g;
            }
        }

        if (candCount == 0) continue;
        int idx = (int)(posVal * candCount);
        if (idx >= candCount) idx = candCount - 1;
        int chosen = candidates[idx];
        if (chosen >= 0 && chosen < numGateways)
            clusterSizes[chosen] += 1;
    }

    // === 2) Contar gateways intermediários (relays) ===
    for (int src = 0; src < numGateways; ++src) {
        int nh = d_nextHop[src];
        if (nh >= 0 && nh < numGateways)
            relaysCount[nh] += 1;
    }

    // === 3) Calcular lifetime ===
    const double E_elec = 50e-9;
    const double E_amp = 10e-12;
    const double E_rx = E_elec * PACKET_SIZE;
    const double E_agg = 5e-9 * PACKET_SIZE;

    double minLifetime = 1e30;
    bool anyValid = false;

    for (int g = 0; g < numGateways; ++g) {
        const NodeGPU& gw = d_nodes[g];

        double e_intra = clusterSizes[g] * (E_rx + E_agg);
        double e_inter = 0.0;
        int nh = d_nextHop[g];

        if (nh == -1) {
            double dx = gw.x - bsx;
            double dy = gw.y - bsy;
            double d_bs = sqrt(dx * dx + dy * dy);
            double E_tx = (E_elec + E_amp * d_bs * d_bs) * PACKET_SIZE;
            e_inter = E_tx;
        }
        else if (nh >= 0 && nh < numGateways) {
            const NodeGPU& next = d_nodes[nh];
            double dx = gw.x - next.x;
            double dy = gw.y - next.y;
            double dgw = sqrt(dx * dx + dy * dy);
            double E_tx = (E_elec + E_amp * dgw * dgw) * PACKET_SIZE;
            int relays = relaysCount[g];
            e_inter = relays * E_rx + (relays + 1) * E_tx;
        }
        else continue;

        double total = e_intra + e_inter;
        if (!(total > 0.0) || !isfinite(total)) continue;

        double lifetime = gw.energy / total;
        if (!isfinite(lifetime) || lifetime < 0.0) lifetime = 0.0;

        anyValid = true;
        if (lifetime < minLifetime) minLifetime = lifetime;
    }

    if (!anyValid || !isfinite(minLifetime) || minLifetime > 1e29)
        minLifetime = 0.0;

    d_fitness[pid] = minLifetime;
}

// atualiza pbest: se fitness maior que pbestFitness, copia posicionamento da partícula para pbest
__global__ void updatePersonalBestKernel_Clust(
    const double* d_positions,
    double* d_pbest,
    const double* d_fitness,
    double* d_pbestFitness,
    int swarmSize,
    int numSensors)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= swarmSize) return;

    double f = d_fitness[pid];
    if (f > d_pbestFitness[pid]) {
        d_pbestFitness[pid] = f;
        // copiar pos -> pbest
        size_t base = (size_t)pid * numSensors;
        for (int s = 0; s < numSensors; ++s) {
            d_pbest[base + s] = d_positions[base + s];
        }
    }
}

// atualiza partícula (vel/pos) usando curand states
__global__ void updateParticlesKernel_Clust(
    double* d_positions,
    double* d_velocities,
    const double* d_pbest,
    const double* d_gbest,
    curandState* d_randStates,
    int swarmSize,
    int numSensors,
    double w,
    double c1,
    double c2)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= swarmSize) return;

    curandState localState = d_randStates[pid];
    size_t base = (size_t)pid * numSensors;

    for (int s = 0; s < numSensors; ++s) {
        int idx = base + s;
        double r1 = curand_uniform_double(&localState);
        double r2 = curand_uniform_double(&localState);

        double vel = d_velocities[idx];
        double pos = d_positions[idx];
        double pbest = d_pbest[idx];
        double gbest = d_gbest[s];

        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos);

        // clamp
        if (vel > 0.5) vel = 0.5;
        if (vel < -0.5) vel = -0.5;

        pos += vel;
        if (pos < 0.0) pos = 0.0;
        if (pos > 1.0) pos = 1.0;

        d_velocities[idx] = vel;
        d_positions[idx] = pos;
    }

    d_randStates[pid] = localState;
}

//////////////////////
// class methods
//////////////////////

ClusteringPSO_CUDA::ClusteringPSO_CUDA(Network& net_,
    const std::vector<int>& nextHopHost_,
    const std::vector<double>& clusterRadiiHost_,
    int swarmSize_, int iterations_)
    : net(net_), nextHopHost(nextHopHost_), clusterRadiiHost(clusterRadiiHost_),
    swarmSize(swarmSize_), iterations(iterations_)
{
    numSensors = net.numSensors;
    numGateways = net.numGateways;
}

ClusteringPSO_CUDA::~ClusteringPSO_CUDA() {
    //freeMemory();
}

void ClusteringPSO_CUDA::allocateMemory() {
    // positions, velocities, pbest: swarmSize * numSensors
    size_t particlesBytes = (size_t)swarmSize * (size_t)numSensors * sizeof(double);
    CUDA_CALL(cudaMalloc(&d_positions, particlesBytes));
    CUDA_CALL(cudaMalloc(&d_velocities, particlesBytes));
    CUDA_CALL(cudaMalloc(&d_pbest, particlesBytes));

    CUDA_CALL(cudaMalloc(&d_pbestFitness, sizeof(double) * swarmSize));
    CUDA_CALL(cudaMalloc(&d_gbest, sizeof(double) * numSensors));
    CUDA_CALL(cudaMalloc(&d_fitness, sizeof(double) * swarmSize));

    CUDA_CALL(cudaMalloc(&d_randStates, sizeof(curandState) * swarmSize));

    // network arrays
    CUDA_CALL(cudaMalloc(&d_nodes, sizeof(NodeGPU) * (numGateways + numSensors)));
    CUDA_CALL(cudaMalloc(&d_nextHop, sizeof(int) * numGateways));
    CUDA_CALL(cudaMalloc(&d_clusterRadii, sizeof(double) * numGateways));

    CUDA_CALL(cudaMalloc(&d_assignment, sizeof(int) * (size_t)swarmSize * numSensors));
    CUDA_CALL(cudaMalloc(&d_clusterSizes, sizeof(int) * (size_t)swarmSize * numGateways));
    CUDA_CALL(cudaMalloc(&d_relaysCount, sizeof(int) * (size_t)numGateways));

}

void ClusteringPSO_CUDA::freeMemory() {
    if (d_positions) CUDA_CALL(cudaFree(d_positions));
    if (d_velocities) CUDA_CALL(cudaFree(d_velocities));
    if (d_pbest) CUDA_CALL(cudaFree(d_pbest));
    if (d_pbestFitness) CUDA_CALL(cudaFree(d_pbestFitness));
    if (d_gbest) CUDA_CALL(cudaFree(d_gbest));
    if (d_fitness) CUDA_CALL(cudaFree(d_fitness));
    if (d_randStates) CUDA_CALL(cudaFree(d_randStates));

    if (d_nodes) CUDA_CALL(cudaFree(d_nodes));
    if (d_nextHop) CUDA_CALL(cudaFree(d_nextHop));
    if (d_clusterRadii) CUDA_CALL(cudaFree(d_clusterRadii));

    if (d_assignment) CUDA_CALL(cudaFree(d_assignment));
    if (d_clusterSizes) CUDA_CALL(cudaFree(d_clusterSizes));
    if (d_relaysCount) CUDA_CALL(cudaFree(d_relaysCount));

}

void ClusteringPSO_CUDA::copyNetworkToDevice() {
    // Cria vetor NodeGPU (host) e copia
    std::vector<NodeGPU> hostNodes;
    hostNodes.reserve(numGateways + numSensors);
    // assume Network::nodes layout: gateways first, then sensors
    for (int i = 0; i < numGateways + numSensors; ++i) {
        const Node& n = net.nodes[i];
        NodeGPU ng(n.x, n.y, n.energy, n.id, n.isGateway ? 1 : 0);
        hostNodes.push_back(ng);
    }
    CUDA_CALL(cudaMemcpy(d_nodes, hostNodes.data(), sizeof(NodeGPU) * hostNodes.size(), cudaMemcpyHostToDevice));

    // nextHop
    std::vector<int> h_next = nextHopHost;
    CUDA_CALL(cudaMemcpy(d_nextHop, h_next.data(), sizeof(int) * numGateways, cudaMemcpyHostToDevice));

    // cluster radii
    std::vector<double> h_radii = clusterRadiiHost;
    CUDA_CALL(cudaMemcpy(d_clusterRadii, h_radii.data(), sizeof(double) * numGateways, cudaMemcpyHostToDevice));
}

void ClusteringPSO_CUDA::initializeParticles() {
    // inicializa pbestFitness com valor muito baixo
    {
        std::vector<double> h_init(swarmSize, -1e300);
        CUDA_CALL(cudaMemcpy(d_pbestFitness, h_init.data(), sizeof(double) * swarmSize, cudaMemcpyHostToDevice));
    }

    // kernel init de posições/velocidades
    dim3 threads(256);
    dim3 blocks(((swarmSize * numSensors) + threads.x - 1) / threads.x);
    initParticlesClustKernel << <blocks, threads >> > (d_positions, d_velocities, swarmSize, numSensors, 123456UL);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // copiar positions iniciais para pbest
    size_t particlesBytes = (size_t)swarmSize * (size_t)numSensors * sizeof(double);
    CUDA_CALL(cudaMemcpy(d_pbest, d_positions, particlesBytes, cudaMemcpyDeviceToDevice));

    // init curand states
    dim3 t2(256);
    dim3 b2((swarmSize + t2.x - 1) / t2.x);
    initCurandKernel << <b2, t2 >> > (d_randStates, 424242UL, swarmSize);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
}

std::vector<double> ClusteringPSO_CUDA::getGBestHost() const {    
    return h_gbest_cache;
}

void ClusteringPSO_CUDA::run() {
    std::cout << "[CUDA][ClusteringPSO] Iniciando...\n";

    allocateMemory();
    copyNetworkToDevice();
    initializeParticles();

    // --- preparar relaysCount (uma vez) a partir de nextHopHost (host) ---
    {
        std::vector<int> h_relays(numGateways, 0);
        for (int src = 0; src < (int)nextHopHost.size(); ++src) {
            int nh = nextHopHost[src];
            if (nh >= 0 && nh < numGateways) h_relays[nh]++;
        }
        CUDA_CALL(cudaMemcpy(d_relaysCount, h_relays.data(),
            sizeof(int) * (size_t)numGateways,
            cudaMemcpyHostToDevice));
    }

    dim3 threads(512);
    dim3 blocksParticles((swarmSize + threads.x - 1) / threads.x);

    // para kernels assign/count (cada thread = particle*sensor)
    size_t totalAssign = (size_t)swarmSize * (size_t)numSensors;
    int blocksAssignInt = (int)((totalAssign + threads.x - 1) / threads.x);
    dim3 blocksAssign(blocksAssignInt);

    std::vector<double> h_bestHistory;
    h_bestHistory.reserve(iterations);

    // controle do melhor global (host)
    double globalBestVal = -1e300;

    // ============================================================
    //   CONFIGURAÇÃO DO LOOP (Thread) - escolha aqui
    // ============================================================

     #define USE_PSO_TARGET_CRITERIA   // loop novo: para por target / timeout / estagnação
    //#define USE_PSO_FIXED_ITER           // loop antigo: iterações fixas (iterations)

    #if !defined(USE_PSO_TARGET_CRITERIA) && !defined(USE_PSO_FIXED_ITER)
        #define USE_PSO_FIXED_ITER
    #endif

// ----------------- parâmetros exclusivos do loop novo -----------------
#if defined(USE_PSO_TARGET_CRITERIA)
    const bool   stopOnTarget = true;             // se true usa targetFitness como critério
    const double targetFitness = 2000.0;           // fitness alvo
    const bool   useMaxIter = true;             // critério alternativo de iterações
    const int    maxIter = 100000;            // número máximo de iterações "esperadas"
    const int    maxAllowedIterations = 200000;        // fallback absoluto
    const double maxWallTimeMs = 1000.0 * 60.0 * 30.0; // 30 minutos
    const int    stagnationLimit = 2000;             // sem melhoria por N iterações -> parar

    int  iter = 0;
    int  lastImprovementIter = 0;
#endif

    auto t0 = std::chrono::high_resolution_clock::now();

    // ============================================================
    //   LOOP PRINCIPAL DO PSO (Thread)
    // ============================================================

#if defined(USE_PSO_TARGET_CRITERIA)

// ============================
//   LOOP NOVO (while true)
// ============================
    while (true) {
        // ---------- (A) Pipeline: assign -> count -> computeFitness ----------
        CUDA_CALL(cudaMemset(d_clusterSizes, 0,
            sizeof(int) * (size_t)swarmSize * (size_t)numGateways));

        assignSensorsKernel << <blocksAssign, threads >> > (
            d_positions, d_assignment, swarmSize, numSensors, numGateways,
            d_nodes, d_clusterRadii);
        CUDA_CALL(cudaGetLastError());

        countClustersKernel << <blocksAssign, threads >> > (
            d_assignment, d_clusterSizes, swarmSize, numSensors, numGateways);
        CUDA_CALL(cudaGetLastError());

        computeFitnessPerParticleKernel << <blocksParticles, threads >> > (
            d_clusterSizes, d_fitness, swarmSize, numGateways,
            d_nodes, d_nextHop, d_relaysCount,
            net.bs.x, net.bs.y, net.gatewayRange);
        CUDA_CALL(cudaGetLastError());

        CUDA_CALL(cudaDeviceSynchronize());

        // ---------- (B) atualizar pbest (cada thread = 1 particle) ----------
        updatePersonalBestKernel_Clust << <blocksParticles, threads >> > (
            d_positions, d_pbest, d_fitness, d_pbestFitness,
            swarmSize, numSensors);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // ---------- (C) host-side argmax sobre PBEST FITNESS e atualização condicional do GBEST ----------
        {
            std::vector<double> h_pbestFitness(swarmSize);
            CUDA_CALL(cudaMemcpy(h_pbestFitness.data(), d_pbestFitness,
                sizeof(double) * (size_t)swarmSize,
                cudaMemcpyDeviceToHost));

            int    bestIndex = 0;
            double bestValIter = h_pbestFitness[0];
            for (int i = 1; i < swarmSize; ++i) {
                if (h_pbestFitness[i] > bestValIter) {
                    bestValIter = h_pbestFitness[i];
                    bestIndex = i;
                }
            }

            if (bestValIter > globalBestVal) {
                globalBestVal = bestValIter;
                lastImprovementIter = iter;

                auto   tNow = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
                gbestTimeline.emplace_back(ms, globalBestVal);

                size_t  bytes = sizeof(double) * (size_t)numSensors;
                double* src = d_pbest + (size_t)bestIndex * (size_t)numSensors;
                CUDA_CALL(cudaMemcpy(d_gbest, src, bytes, cudaMemcpyDeviceToDevice));
            }

            // armazenar o melhor global atual (mesmo que não tenha mudado)
            h_bestHistory.push_back(globalBestVal);
        }

        // ---------- Critérios de parada (após atualização do GBEST) ----------
        {
            // 1) Se alcançou o target (caso esteja ativado)
            if (stopOnTarget && globalBestVal >= targetFitness) {
                auto   tNow = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
                printf("Parando: atingiu targetFitness (%.3f) em %.3f ms (iter=%d)\n",
                    globalBestVal, ms, iter);
                break;
            }

            // 2) Critério clássico de iterações
            if (useMaxIter && iter >= maxIter) {
                auto   tNow = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
                printf("Parando: atingiu maxIter (%d) em %.3f ms\n", iter, ms);
                break;
            }

            // 3) Fallback absoluto por iterações
            if (iter >= maxAllowedIterations) {
                auto   tNow = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
                printf("Parando por segurança: maxAllowedIterations (%d) alcançado em %.3f ms\n",
                    iter, ms);
                break;
            }

            // 4) Fallback por tempo de wall-clock
            {
                auto   tNow = std::chrono::high_resolution_clock::now();
                double elapsedMs = std::chrono::duration<double, std::milli>(tNow - t0).count();
                if (elapsedMs >= maxWallTimeMs) {
                    printf("Parando por segurança: timeout de %.0f ms alcançado (%.3f ms)\n",
                        maxWallTimeMs, elapsedMs);
                    break;
                }
            }

            // 5) Estagnação (nenhuma melhoria por N iterações)
            if ((iter - lastImprovementIter) >= stagnationLimit) {
                auto   tNow = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
                printf("Parando por estagnação: sem melhoria por %d iters (iter=%d) em %.3f ms. Fitness: %.2f\n", stagnationLimit, iter, ms, globalBestVal);
                break;
            }
        }

        // ---------- (D) atualizar partículas (vel/pos) ----------
        updateParticlesKernel_Clust << <blocksParticles, threads >> > (
            d_positions, d_velocities, d_pbest, d_gbest, d_randStates,
            swarmSize, numSensors, 0.7968, 1.4962, 1.4962);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        ++iter;
    }

#elif defined(USE_PSO_FIXED_ITER)

// ============================
//   LOOP ANTIGO (for fixo)
// ============================
    for (int iter = 0; iter < iterations; ++iter) {
        // ---------- (A) Pipeline: assign -> count -> computeFitness ----------
        CUDA_CALL(cudaMemset(d_clusterSizes, 0,
            sizeof(int) * (size_t)swarmSize * (size_t)numGateways));

        assignSensorsKernel << <blocksAssign, threads >> > (
            d_positions, d_assignment, swarmSize, numSensors, numGateways,
            d_nodes, d_clusterRadii);
        CUDA_CALL(cudaGetLastError());

        countClustersKernel << <blocksAssign, threads >> > (
            d_assignment, d_clusterSizes, swarmSize, numSensors, numGateways);
        CUDA_CALL(cudaGetLastError());

        computeFitnessPerParticleKernel << <blocksParticles, threads >> > (
            d_clusterSizes, d_fitness, swarmSize, numGateways,
            d_nodes, d_nextHop, d_relaysCount,
            net.bs.x, net.bs.y, net.gatewayRange);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // ---------- (B) atualizar pbest ----------
        updatePersonalBestKernel_Clust << <blocksParticles, threads >> > (
            d_positions, d_pbest, d_fitness, d_pbestFitness,
            swarmSize, numSensors);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // ---------- (C) host-side argmax + gbest ----------
        {
            std::vector<double> h_pbestFitness(swarmSize);
            CUDA_CALL(cudaMemcpy(h_pbestFitness.data(), d_pbestFitness,
                sizeof(double) * (size_t)swarmSize,
                cudaMemcpyDeviceToHost));

            int    bestIndex = 0;
            double bestValIter = h_pbestFitness[0];
            for (int i = 1; i < swarmSize; ++i) {
                if (h_pbestFitness[i] > bestValIter) {
                    bestValIter = h_pbestFitness[i];
                    bestIndex = i;
                }
            }

            if (bestValIter > globalBestVal) {
                globalBestVal = bestValIter;

                auto   tNow = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
                gbestTimeline.emplace_back(ms, globalBestVal);

                size_t  bytes = sizeof(double) * (size_t)numSensors;
                double* src = d_pbest + (size_t)bestIndex * (size_t)numSensors;
                CUDA_CALL(cudaMemcpy(d_gbest, src, bytes, cudaMemcpyDeviceToDevice));
            }

            h_bestHistory.push_back(globalBestVal);
        }

        // ---------- (D) atualizar partículas ----------
        updateParticlesKernel_Clust << <blocksParticles, threads >> > (
            d_positions, d_velocities, d_pbest, d_gbest, d_randStates,
            swarmSize, numSensors, 0.7968, 1.4962, 1.4962);
        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());
    }

#endif // fim escolha de loop

    // ============================================================
    // PÓS-LOOP: export, decode, etc. (seu código original)
    // ============================================================
    auto tEnd = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(tEnd - t0).count();

    // export converge
    {
        std::ofstream csv("pso_convergence_gpu_clustering.csv");
        csv << "iteration,best_fitness\n";
        for (int i = 0; i < (int)h_bestHistory.size(); ++i)
            csv << i << "," << h_bestHistory[i] << "\n";
        csv.close();
    }

    {
        std::ofstream out("gbest_timeline_gpu.csv");
        out << "time_ms,fitness\n";
        for (auto& p : gbestTimeline)
            out << p.first << "," << p.second << "\n";
        out << "END," << total_ms << "\n";
        out.close();
    }

    // copiar gbest para host cache
    if (d_gbest) {
        h_gbest_cache.resize(numSensors);
        CUDA_CALL(cudaMemcpy(h_gbest_cache.data(), d_gbest,
            sizeof(double) * (size_t)numSensors,
            cudaMemcpyDeviceToHost));
    }

    std::vector<int> sensorOffsets(numSensors + 1);
    std::vector<int> sensorAdj;
    sensorOffsets[0] = 0;
    sensorAdj.reserve(numSensors * 8);

    // clusterRadiiHost já existe no seu objeto
    for (int s = 0; s < numSensors; ++s) {
        int count = 0;
        const Node& sensor = net.nodes[numGateways + s];

        for (int g = 0; g < numGateways; ++g) {
            const Node& gw = net.nodes[g];
            double dx = gw.x - sensor.x;
            double dy = gw.y - sensor.y;
            double d2 = dx * dx + dy * dy;
            double r = clusterRadiiHost[g];
            if (d2 <= r * r) {
                sensorAdj.push_back(g);
                count++;
            }
        }

        sensorOffsets[s + 1] = sensorOffsets[s] + count;
    }

    // ============================================================
    // DECODE FINAL DO CLUSTERING USANDO O GBEST DO PSO
    // ============================================================
    std::vector<int> assignmentGPU(numSensors);

    for (int s = 0; s < numSensors; ++s) {
        int start = sensorOffsets[s];
        int end = sensorOffsets[s + 1];
        int deg = end - start;

        int assigned = -1;
        if (deg > 0) {
            int pick = (int)(h_gbest_cache[s] * deg);
            if (pick >= deg) pick = deg - 1;
            assigned = sensorAdj[start + pick];
        }

        assignmentGPU[s] = assigned;
    }

    // ============================================================
    // EXPORTAÇÃO
    // ============================================================
    exportNetworkAndLinksToCSV(
        net,
        "gpu_network.csv",
        nextHopHost,
        assignmentGPU,
        clusterRadiiHost
    );

    freeMemory();
}


#endif // USE_CUDA
