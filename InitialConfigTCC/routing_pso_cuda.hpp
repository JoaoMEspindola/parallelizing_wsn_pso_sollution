#ifndef ROUTING_PSO_CUDA_HPP
#define ROUTING_PSO_CUDA_HPP

#include "network.hpp"
#include "energy_gpu.hpp"
#include <vector>
#include <cuda_runtime.h>
#include "compact_graph.hpp"
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    fprintf(stderr, "Erro CUDA: %s (%s:%d)\n", cudaGetErrorString(x), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); } } while(0)

class RoutingPSO_CUDA {
public:
    RoutingPSO_CUDA(Network& net_, int swarmSize_, int iterations_);
    void run();
    std::vector<double> getApproxLifetime() const;
    std::vector<double> getGBestHost() const;              // copia gbest da GPU para host
    std::vector<int> decodeRoutingGBestToNextHop() const;  // converte gbest em nextHop

private:
    Network& net;
    int swarmSize, iterations, numGateways;

    curandState* d_randStates; // Um gerador por thread

    double* d_pbest;
    double* d_gbest;

    double* d_positions;
    double* d_velocities;
    double* d_fitness;  // armazenará o fitness de cada partícula
    double* d_pbestFitness;
    double* d_globalBestFitness;
    // Dados do grafo compactado
    CompactGraphDevice graphDev;
    bool graphAllocated = false; // controle de alocação
    // host cache para gbest (copiado antes do freeMemory)
    mutable std::vector<double> h_gbest_cache;
    mutable bool h_gbest_cached = false;


    void allocateMemory();
    void initializeParticles();
    void freeMemory();

};

#endif
