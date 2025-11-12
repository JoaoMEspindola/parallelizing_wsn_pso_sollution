#ifndef CLUSTERING_PSO_CUDA_HPP
#define CLUSTERING_PSO_CUDA_HPP

#include <vector>
#include "network.hpp"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::cerr << "Erro CUDA: " << cudaGetErrorString(e) \
              << " (" << __FILE__ << ":" << __LINE__ << ")\n"; exit(EXIT_FAILURE); } } while(0)

class ClusteringPSO_CUDA {
public:
    ClusteringPSO_CUDA(Network& net_,
        const std::vector<int>& nextHopHost_,
        const std::vector<double>& clusterRadiiHost_,
        int swarmSize_, int iterations_);

    ~ClusteringPSO_CUDA();

    // Executa PSO paralelo e exporta "pso_convergence_gpu_clustering.csv"
    void run();

    // Retorna o vetor gbest do host após a execução
    std::vector<double> getGBestHost() const;

private:
    // ======= Dados principais =======
    Network& net;
    const std::vector<int>& nextHopHost;
    const std::vector<double>& clusterRadiiHost;

    int swarmSize, iterations;
    int numSensors, numGateways;

    std::vector<double> h_gbest_cache;

    // ======= Ponteiros no device =======

    // --- PSO data ---
    double* d_positions = nullptr;     // swarmSize * numSensors
    double* d_velocities = nullptr;    // swarmSize * numSensors
    double* d_pbest = nullptr;         // swarmSize * numSensors
    double* d_pbestFitness = nullptr;  // swarmSize
    double* d_gbest = nullptr;         // numSensors
    double* d_fitness = nullptr;       // swarmSize
    curandState* d_randStates = nullptr;

    // --- Network data ---
    NodeGPU* d_nodes = nullptr;        // numGateways + numSensors
    int* d_nextHop = nullptr;          // numGateways
    double* d_clusterRadii = nullptr;  // numGateways

    // --- Auxiliares para kernels paralelos ---
    int* d_assignment = nullptr;       // swarmSize * numSensors
    int* d_clusterSizes = nullptr;     // swarmSize * numGateways
    int* d_relaysCount = nullptr;      // numGateways (pré-calculado)

    // ======= Métodos auxiliares =======
    void allocateMemory();
    void freeMemory();
    void initializeParticles();
    void copyNetworkToDevice();
};

#endif // USE_CUDA
#endif // CLUSTERING_PSO_CUDA_HPP
