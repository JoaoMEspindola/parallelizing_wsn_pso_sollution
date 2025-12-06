#include "routing_pso_cuda.hpp"
#include "cuda_common_kernels.hpp"

#include "utils.hpp"
#include "energy.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include "compact_graph.hpp"
#include <fstream>
#include <algorithm>

// -----------------------------------------------
// Kernel de inicialização
__global__ void initParticlesKernel(double* positions, double* velocities, int numParticles, int numGateways) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles * numGateways) return;

    positions[idx] = (double)(idx % 1000) / 1000.0;
    velocities[idx] = ((double)((idx * 37) % 1000) / 1000.0) - 0.5;
}

// ===========================
// Funções auxiliares GPU
// ===========================

// =============================================================
// Kernel: avalia cada partícula (thread = 1 partícula)
// =============================================================
// ==========================================================
// Kernel principal: cada thread avalia uma partícula do enxame
// ==========================================================

__global__ void evaluateParticlesKernel(
    const double* d_positions,
    double* d_fitness,
    int swarmSize,
    int numGateways,
    const NodeGPU* d_nodes,
    const int* d_offsets,
    const int* d_adjacency,
    double bsx,
    double bsy,
    double gatewayRange)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= swarmSize) return;

    // --- aloca dinamicamente arrays por thread no heap do device
    int* nextHop = new int[numGateways];
    int* recvCount = new int[numGateways];
    if (!nextHop || !recvCount) {
        // falha de alocação: marca fitness 0 e sai
        if (pid == 0) printf("[evaluateParticlesKernel] allocation failed for numGateways=%d\n", numGateways);
        if (nextHop) delete[] nextHop;
        if (recvCount) delete[] recvCount;
        d_fitness[pid] = 0.0;
        return;
    }

    // inicializa arrays
    for (int g = 0; g < numGateways; ++g) { nextHop[g] = -1; recvCount[g] = 0; }

    // 1) decodifica nextHop para esta partícula (sem compartilhar)
    // cuidado com indexing: pid * numGateways
    size_t baseIdx = (size_t)pid * (size_t)numGateways;
    for (int g = 0; g < numGateways; ++g) {
        double posVal = d_positions[baseIdx + g];
        int start = d_offsets[g];
        int end = d_offsets[g + 1];
        int degree = end - start;

        int selected = -1;
        if (degree > 0) {
            int idx = min((int)(posVal * degree), degree - 1);
            selected = d_adjacency[start + idx];
            // garantir que selected é gateway válido
            if (selected < 0 || selected >= numGateways) selected = -1;
        }
        nextHop[g] = selected;
    }

    // 2) calcula recvCount localmente (nested loops)
    for (int s = 0; s < numGateways; ++s) {
        int nh = nextHop[s];
        if (nh >= 0 && nh < numGateways) ++recvCount[nh];
    }

    // 3) calcula minLifetime (igual antes)
    const double E_elec = 50e-9;
    const double E_amp = 10e-12;
    const double k = 4000.0;

    double minLifetime = 1e30;
    bool anyValid = false;

    for (int g = 0; g < numGateways; ++g) {
        const NodeGPU& nodeG = d_nodes[g];
        int nh = nextHop[g];

        double dist;
        if (nh == -1) {
            dist = hypot(nodeG.x - bsx, nodeG.y - bsy);
        }
        else {
            const NodeGPU& next = d_nodes[nh];
            dist = hypot(nodeG.x - next.x, nodeG.y - next.y);
        }

        if (nh != -1 && dist > gatewayRange) { minLifetime = 0.0; continue; }

        double E_tx = (E_elec + E_amp * dist * dist) * k;
        double E_rx = E_elec * k;
        int n_recv = recvCount[g];

        double E_total = (n_recv + 1) * E_tx + n_recv * E_rx;
        if (E_total <= 0.0 || !isfinite(E_total)) continue;

        double lifetime = nodeG.energy / E_total;
        if (!isfinite(lifetime) || lifetime < 0.0) lifetime = 0.0;

        anyValid = true;
        if (lifetime < minLifetime) minLifetime = lifetime;
    }

    if (!isfinite(minLifetime) || minLifetime > 1e29 || !anyValid) minLifetime = 0.0;
    d_fitness[pid] = minLifetime;

    // libera memória
    delete[] nextHop;
    delete[] recvCount;
}



// ==========================================================
// Atualiza posição e velocidade das partículas (núcleo do PSO)
// ==========================================================
__global__ void updateParticlesKernel(
    double* d_positions,
    double* d_velocities,
    const double* d_pbest,
    const double* d_gbest,
    curandState* d_randStates,
    int swarmSize,
    int numGateways,
    double w, double c1, double c2)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= swarmSize) return;

    curandState localState = d_randStates[pid];

    for (int g = 0; g < numGateways; ++g) {
        int idx = pid * numGateways + g;
        double r1 = curand_uniform(&localState);
        double r2 = curand_uniform(&localState);

        double vel = w * d_velocities[idx] + c1 * r1 * (d_pbest[idx] - d_positions[idx]) + c2 * r2 * (d_gbest[g] - d_positions[idx]);
        vel = fmin(fmax(vel, -0.5), 0.5);
        d_velocities[idx] = vel;

        double pos = d_positions[idx] + vel;
        d_positions[idx] = fmin(fmax(pos, 0.0), 1.0);
    }

    d_randStates[pid] = localState;
}

// ==========================================================
// Atualiza o melhor pessoal (pbest) de cada partícula
// ==========================================================
__global__ void updatePersonalBestKernel(
    const double* d_positions,
    double* d_pbest,
    const double* d_fitness,
    double* d_pbestFitness,
    int swarmSize,
    int numGateways)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= swarmSize) return;

    double fit = d_fitness[pid];
    if (fit > d_pbestFitness[pid]) {
        d_pbestFitness[pid] = fit;
        for (int g = 0; g < numGateways; ++g)
            d_pbest[pid * numGateways + g] = d_positions[pid * numGateways + g];
    }
}



// ==========================================================
// Atualiza o melhor global (gbest) via redução
// ==========================================================
__global__ void updateGlobalBestKernel(
    const double* d_fitness,
    const double* d_pbest,
    double* d_gbest,
    double* d_globalBestFitness,
    int swarmSize,
    int numGateways)
{
    extern __shared__ unsigned char smem[];
    double* s_fitness = reinterpret_cast<double*>(smem);
    int* s_index = reinterpret_cast<int*>(s_fitness + blockDim.x);

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double myFit = (i < swarmSize) ? d_fitness[i] : -1e300;
    s_fitness[tid] = myFit;
    s_index[tid] = i;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && s_fitness[tid + offset] > s_fitness[tid]) {
            s_fitness[tid] = s_fitness[tid + offset];
            s_index[tid] = s_index[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int bestIdx = s_index[0];
        double bestFit = s_fitness[0];

        if (bestIdx >= 0 && bestFit > *d_globalBestFitness) {
            *d_globalBestFitness = bestFit;
            const double* src = d_pbest + (size_t)bestIdx * numGateways;
            for (int j = 0; j < numGateways; ++j)
                d_gbest[j] = src[j];
        }
    }
}



// -----------------------------------------------
// Construtor
RoutingPSO_CUDA::RoutingPSO_CUDA(Network& net_, int swarmSize_, int iterations_)
    : net(net_), swarmSize(swarmSize_), iterations(iterations_) {
    numGateways = net.numGateways;
}

// -----------------------------------------------
// Método principal
void RoutingPSO_CUDA::run() {
    std::cout << "[CUDA] Inicializando PSO paralelo...\n";

    // === CONVERTER REDE PARA LISTA COMPACTA ===
    CompactGraphHost hostGraph = buildCompactGraph(net);

    // === COPIAR GRAFO PARA GPU ===
    graphDev.totalNodes = static_cast<int>(hostGraph.h_nodes.size());
    graphDev.totalEdges = static_cast<int>(hostGraph.h_adjacency.size());

    CUDA_CALL(cudaMalloc(&graphDev.d_nodes, sizeof(NodeGPU) * graphDev.totalNodes));
    CUDA_CALL(cudaMalloc(&graphDev.d_offsets, sizeof(int) * hostGraph.h_offsets.size()));
    CUDA_CALL(cudaMalloc(&graphDev.d_adjacency, sizeof(int) * graphDev.totalEdges));

    CUDA_CALL(cudaMemcpy(graphDev.d_nodes, hostGraph.h_nodes.data(),
        sizeof(NodeGPU) * graphDev.totalNodes, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemcpy(graphDev.d_offsets, hostGraph.h_offsets.data(),
        sizeof(int) * hostGraph.h_offsets.size(), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemcpy(graphDev.d_adjacency, hostGraph.h_adjacency.data(),
        sizeof(int) * graphDev.totalEdges, cudaMemcpyHostToDevice));

    graphAllocated = true;

    // === ALOCA E INICIALIZA PARTÍCULAS ===
    allocateMemory();
    initializeParticles();

    // Inicializa fitness muito baixo
    {
        std::vector<double> h_initPbestFitness(swarmSize, -1e300);
        CUDA_CALL(cudaMemcpy(d_pbestFitness, h_initPbestFitness.data(),
            sizeof(double) * swarmSize, cudaMemcpyHostToDevice));
    }

    // === Inicializar geradores de números aleatórios ===
    dim3 threads(256);
    dim3 blocks((swarmSize + threads.x - 1) / threads.x);

    CUDA_CALL(cudaMemset(d_randStates, 0, sizeof(curandState) * swarmSize));
    initCurandKernel<<<blocks, threads>>>(d_randStates, 1234UL, swarmSize);
    CUDA_CALL(cudaDeviceSynchronize());

    // === Histórico do melhor fitness ===
    std::vector<double> h_bestHistory;
    h_bestHistory.reserve(iterations);

    // === LOOP PRINCIPAL ===

    for (int it = 0; it < iterations; ++it) {
        // (A) Avaliar o fitness de todas as partículas
        CUDA_CALL(cudaMemset(d_fitness, 0, sizeof(double) * swarmSize));
        int sharedMemSize = 2 * numGateways * sizeof(int);
        evaluateParticlesKernel << <blocks, threads, sharedMemSize >> > (
            d_positions,
            d_fitness,
            swarmSize,
            numGateways,
            graphDev.d_nodes,
            graphDev.d_offsets,
            graphDev.d_adjacency,
            net.bs.x,
            net.bs.y,
            net.gatewayRange
            );
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaGetLastError());

        // (B) Atualizar pbest
        updatePersonalBestKernel<<<blocks, threads>>>(
            d_positions, d_pbest, d_fitness, d_pbestFitness,
            swarmSize, numGateways);
        CUDA_CALL(cudaGetLastError());

        // copiar pBest e fitness para host
        std::vector<double> h_pbest(swarmSize * numGateways);
        std::vector<double> h_pbestFitness(swarmSize);

        CUDA_CALL(cudaMemcpy(h_pbest.data(), d_pbest, sizeof(double) * swarmSize * numGateways, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_pbestFitness.data(), d_pbestFitness, sizeof(double) * swarmSize, cudaMemcpyDeviceToHost));

        // encontrar gBest seguro
        double bestFit = -1.0;
        int bestIdx = -1;
        for (int i = 0; i < swarmSize; ++i) {
            if (h_pbestFitness[i] > bestFit) {
                bestFit = h_pbestFitness[i];
                bestIdx = i;
            }
        }

        // atualizar gBest na GPU
        if (bestIdx >= 0) {
            CUDA_CALL(cudaMemcpy(d_gbest, h_pbest.data() + bestIdx * numGateways, sizeof(double) * numGateways, cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(d_globalBestFitness, &bestFit, sizeof(double), cudaMemcpyHostToDevice));
        }


        // (C) Atualizar gbest (AGORA NA GPU!)
        sharedMemSize = threads.x * (sizeof(double) + sizeof(int));
        updateGlobalBestKernel << <1, threads, sharedMemSize >> > (
            d_fitness,
            d_pbest,
            d_gbest,
            d_globalBestFitness,  // <<< NOVO PARÂMETRO AQUI
            swarmSize,
            numGateways
            );

        CUDA_CALL(cudaGetLastError());
        CUDA_CALL(cudaDeviceSynchronize());

        // (D) Atualizar posições e velocidades
        updateParticlesKernel<<<blocks, threads>>>(
            d_positions, d_velocities, d_pbest, d_gbest,
            d_randStates, swarmSize, numGateways,
            0.7968, 1.4962, 1.4962);
        CUDA_CALL(cudaGetLastError());

        CUDA_CALL(cudaDeviceSynchronize());

        // (E) Copiar melhor fitness só para salvar curva
        std::vector<double> h_fitness(swarmSize);
        CUDA_CALL(cudaMemcpy(h_fitness.data(), d_fitness,
            sizeof(double) * swarmSize, cudaMemcpyDeviceToHost));
        h_bestHistory.push_back(bestFit);
    }

    // === Exportar curva ===
    std::ofstream csv("pso_convergence_gpu.csv");
    csv << "iteration,best_fitness\n";
    for (int i = 0; i < h_bestHistory.size(); ++i)
        csv << i << "," << h_bestHistory[i] << "\n";
    csv.close();

    if (d_gbest) {
        h_gbest_cache.assign(numGateways, 0.0);
        CUDA_CALL(cudaMemcpy(h_gbest_cache.data(), d_gbest, sizeof(double) * numGateways, cudaMemcpyDeviceToHost));
        h_gbest_cached = true;
    }

    freeMemory();
    std::cout << "[CUDA] Execução concluída.\n";
}



// -----------------------------------------------
// Métodos auxiliares
void RoutingPSO_CUDA::allocateMemory() {
    CUDA_CALL(cudaMalloc(&d_positions, sizeof(double) * swarmSize * numGateways));
    CUDA_CALL(cudaMalloc(&d_velocities, sizeof(double) * swarmSize * numGateways));
    CUDA_CALL(cudaMalloc(&d_pbest, sizeof(double) * swarmSize * numGateways));
    CUDA_CALL(cudaMalloc(&d_gbest, sizeof(double) * numGateways));
    CUDA_CALL(cudaMalloc(&d_pbestFitness, sizeof(double) * swarmSize));
    CUDA_CALL(cudaMalloc(&d_fitness, sizeof(double) * swarmSize));
    CUDA_CALL(cudaMalloc(&d_randStates, sizeof(curandState) * swarmSize));
    CUDA_CALL(cudaMalloc(&d_globalBestFitness, sizeof(double)));
    double initVal = -1e300;
    CUDA_CALL(cudaMemcpy(d_globalBestFitness, &initVal, sizeof(double), cudaMemcpyHostToDevice));

}


void RoutingPSO_CUDA::initializeParticles() {
    dim3 threads(256);
    dim3 blocks((swarmSize * numGateways + threads.x - 1) / threads.x);
    initParticlesKernel << <blocks, threads >> > (d_positions, d_velocities, swarmSize, numGateways);
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(d_pbest, d_positions,
        sizeof(double) * swarmSize * numGateways, cudaMemcpyDeviceToDevice));

    // Inicializa gbest com zeros (vamos usar depois)
    std::vector<double> h_gbest(numGateways, 0.5);
    CUDA_CALL(cudaMemcpy(d_gbest, h_gbest.data(),
        sizeof(double) * numGateways, cudaMemcpyHostToDevice));

}

void RoutingPSO_CUDA::freeMemory() {

    if (graphAllocated) {
        CUDA_CALL(cudaFree(graphDev.d_nodes));
        CUDA_CALL(cudaFree(graphDev.d_offsets));
        CUDA_CALL(cudaFree(graphDev.d_adjacency));
        graphAllocated = false;
    }

    CUDA_CALL(cudaFree(d_positions));
    CUDA_CALL(cudaFree(d_velocities));
    CUDA_CALL(cudaFree(d_fitness));

    CUDA_CALL(cudaFree(d_pbest));
    CUDA_CALL(cudaFree(d_gbest));
    CUDA_CALL(cudaFree(d_randStates));
    CUDA_CALL(cudaFree(d_globalBestFitness));
}

// -----------------------------------------------
// Placeholder para compatibilidade com o main
std::vector<double> RoutingPSO_CUDA::getApproxLifetime() const {
    // Apenas placeholder para evitar erro
    return std::vector<double>(net.numGateways, 10.0);
}

// ======================================================
// === GET / DECODE FUNCTIONS ===========================
// ======================================================

// Copia d_gbest (double[numGateways]) da GPU para host
std::vector<double> RoutingPSO_CUDA::getGBestHost() const {
    // Se já estivermos com cache host preenchido, retorne-o
    if (h_gbest_cached && (int)h_gbest_cache.size() == numGateways) {
        return h_gbest_cache;
    }

    // Caso contrário, tente copiar diretamente da GPU (comportamento antigo)
    std::vector<double> h_gbest(numGateways, 0.0);
    if (d_gbest) {
        CUDA_CALL(cudaMemcpy(h_gbest.data(), d_gbest, sizeof(double) * numGateways, cudaMemcpyDeviceToHost));
    }
    return h_gbest;
}


// Decodifica gbest (posições [0..1]) para vetor de nextHop (índices de gateways ou -1)
std::vector<int> RoutingPSO_CUDA::decodeRoutingGBestToNextHop() const {
    std::vector<double> h_gbest = getGBestHost();
    std::vector<int> nextHop(numGateways, -1);

    for (int g = 0; g < numGateways; ++g) {
        // obtém gateways vizinhos válidos do grafo
        std::vector<int> candidates = net.getNextHopCandidates(g);
        if (candidates.empty()) continue;

        double posVal = h_gbest[g];
        int idx = (int)(posVal * candidates.size());
        if (idx >= (int)candidates.size()) idx = (int)candidates.size() - 1;
        nextHop[g] = candidates[idx];
    }

    return nextHop;
}
