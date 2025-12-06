#pragma once
#ifndef ROUTING_PSO_BLOCK_CUDA_HPP
#define ROUTING_PSO_BLOCK_CUDA_HPP

#include "cuda_common_kernels.hpp"
#include "utils.hpp"
#include "energy.hpp"
#include "compact_graph.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>


#define CUDA_CALL(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "Erro CUDA: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(EXIT_FAILURE); } } while(0)

/*
  RoutingPSO_Block_CUDA:
  - Cada bloco representa uma "revoada" (swarm local).
  - Cada thread dentro do bloco representa uma partícula dessa revoada.
  - O kernel de avaliação calcula fitness por partícula (thread).
  - Depois, cada bloco reduz para encontrar o melhor particle-in-block (block-best)
    e copia o pbest correspondente para d_blockGbest (uma candidata por bloco).
  - No host: fazemos uma redução entre block-bests para obter o gbest global
    (poderia ser feito em GPU também; deixei host para simplicidade/robustez).
*/

class RoutingPSO_Block_CUDA {
public:
    // numRevoadas: quantas revoadas distintas (i.e. número de blocos) — default 8
    RoutingPSO_Block_CUDA(Network& net_, int swarmSize_, int iterations_, int numRevoadas_);
    ~RoutingPSO_Block_CUDA();

    void run();

    // copia gbest da GPU para host (double[numGateways])
    std::vector<double> getGBestHost() const;

    // decodifica o gbest (double per gateway ∈ [0,1]) em nextHops (igual ao CPU)
    std::vector<int> decodeRoutingGBestToNextHop() const;

private:
    Network& net;
    int swarmSize;
    int iterations;
    int numGateways;
    int numRevoadas;          // número de blocos (revoadas)
    int particlesPerRevoada;  // threads per block (ppr) - computed

    // Device pointers (similar to sua versão atual)
    double* d_positions = nullptr;      // swarmSize * numGateways
    double* d_velocities = nullptr;     // swarmSize * numGateways
    double* d_pbest = nullptr;          // swarmSize * numGateways
    double* d_pbestFitness = nullptr;   // swarmSize
    double* d_fitness = nullptr;        // swarmSize
    double* d_gbest = nullptr;          // numGateways (global best)
    curandState* d_randStates = nullptr;

    // block-level (per revoada) storage
    double* d_blockGbest = nullptr;     // numRevoadas * numGateways
    double* d_blockBestFitness = nullptr;// numRevoadas

    // compact graph device (if you want to use adjacency)
    CompactGraphDevice graphDev;
    bool graphAllocated = false;

    // helpers
    void allocateMemory();
    void freeMemory();
    void initializeParticles();

    // utility
    std::vector<double> h_blockBestBuffer; // host buffer to receive block bests
};

#endif // ROUTING_PSO_BLOCK_CUDA_HPP
