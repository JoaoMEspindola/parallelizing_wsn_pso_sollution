#ifndef CLUSTERING_PSO_CUDA_BLOCK_HPP
#define CLUSTERING_PSO_CUDA_BLOCK_HPP

#ifdef USE_CUDA

#include <vector>
#include "network.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "node_gpu.hpp"
#include "export.hpp"
#include "cuda_common_kernels.hpp"
#include "energy.hpp"
#include "utils.hpp"

#define CUDA_CALL_BLOCK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    std::cerr<<"CUDA ERROR: "<<cudaGetErrorString(e)<<" ("<<__FILE__<<":"<<__LINE__<<")\n"; exit(EXIT_FAILURE);} } while(0)

class ClusteringPSO_CUDA_Block {
public:
    // swarmPerBlock = número de partículas por bloco (tipicamente 32)
    // numBlocks = número de blocos (tipicamente número de SMs, ex: 6)
    ClusteringPSO_CUDA_Block(Network& net_,
        const std::vector<int>& nextHopHost_,
        const std::vector<double>& clusterRadiiHost_,
        int swarmPerBlock_ = 32,
        int numBlocks_ = 6,
        int iterations_ = 100);

    ~ClusteringPSO_CUDA_Block();

    // executa PSO paralelo (gera um CSV de convergência por bloco)
    void run();

    // retorna o gbest combinado (host) - tamanho = numSensors
    std::vector<double> getGBestHost() const;

    // retorna histórico (host) do best por iteração (merged best)
    const std::vector<double>& getHistory() const { return h_bestHistory; }

private:
    Network& net;
    const std::vector<int>& nextHopHost;
    const std::vector<double>& clusterRadiiHost;

    int swarmPerBlock;
    int numBlocks;
    int totalParticles; // swarmPerBlock * numBlocks
    int iterations;

    int numSensors;
    int numGateways;

    // device pointers
    double* d_positions = nullptr;   // totalParticles * numSensors
    double* d_velocities = nullptr;  // totalParticles * numSensors
    double* d_pbest = nullptr;       // totalParticles * numSensors
    double* d_gbest_blocks = nullptr; // numBlocks * numSensors (block-local best)
    double* d_blockBestFitness = nullptr; // numBlocks
    double* d_pbestFitness = nullptr;     // totalParticles
    curandState* d_randStates = nullptr;
    double* d_fitness;
    int* __restrict__ d_relaysCount;
    // --- fields para sensor->gateway adjacency (host -> device) ---
    int* d_sensorOffsets = nullptr;   // length = numSensors + 1
    int* d_sensorAdj = nullptr;   // flattened adjacency
    bool sensorAdjAllocated = false;
    int bestBlockIdx = -1;

    // network data
    NodeGPU* d_nodes = nullptr;
    int* d_nextHop = nullptr;
    double* d_clusterRadii = nullptr;

    // host caches
    std::vector<double> h_gbest_cache; // final best merged
    std::vector<double> h_bestHistory; // merged best per iteration

    // helpers
    void allocateMemory();
    void freeMemory();
    void copyNetworkToDevice();
    void initializeParticles(unsigned long seed = 424242UL);
    void mergeBlockGBestToHostAndSelect(); // helper usado no run()

    // non-copyable
    ClusteringPSO_CUDA_Block(const ClusteringPSO_CUDA_Block&) = delete;
    ClusteringPSO_CUDA_Block& operator=(const ClusteringPSO_CUDA_Block&) = delete;
};

#endif // USE_CUDA

#endif // CLUSTERING_PSO_CUDA_BLOCK_HPP
