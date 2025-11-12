#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "network.hpp"
#include "routing_pso.hpp"
#include "cluster_radius.hpp"
#include "clustering_pso.hpp"
#include "simulation.hpp"
#include "utils.hpp"
#include "export.cpp"
#ifdef USE_CUDA
#include "routing_pso_cuda.hpp"
#include "clustering_pso_cuda.hpp"
#endif

// ======================================================
// === Função auxiliar: medir tempo de execução ========
// ======================================================
template<typename Func>
double measureTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ======================================================
// === Função principal ================================
// ======================================================
int main() {
    // === PARÂMETROS BASEADOS NO ARTIGO ===
    const int numSensors = 600;
    const int numGateways = 60;
    const double areaWidth = 500.0, areaHeight = 500.0;
    const int swarmSize = 50;
    const int iterations = 100;

    // === GERA REDE ===
    Network net(numSensors, numGateways, areaWidth, areaHeight);
    net.generate();
    exportNetworkAndLinksToCSV(net, "network_full.csv");

    std::cout << "Rede gerada com " << numSensors << " sensores e "
        << numGateways << " gateways.\n";

    std::ofstream csv("speedup_summary.csv");
    csv << "Stage,CPU_Time(s),GPU_Time(s),Speedup\n";

    // ======================================================
    // === ROTEAMENTO (CPU e GPU) ===========================
    // ======================================================
    std::cout << "\n========== ROTEAMENTO ==========\n";

    std::vector<int> nextHopCPU;
    std::vector<int> nextHopGPU;
    std::vector<double> approxLifetimeCPU;
    std::vector<double> approxLifetimeGPU;

    double cpuTime_r = measureTime([&]() {
        RoutingPSO rPSO_cpu(net, swarmSize, iterations);
        nextHopCPU = rPSO_cpu.optimizeRouting();
        approxLifetimeCPU = rPSO_cpu.getApproxLifetime();
        });

#ifdef USE_CUDA
    double gpuTime_r = measureTime([&]() {
        RoutingPSO_CUDA rPSO_gpu(net, swarmSize, iterations);
        rPSO_gpu.run();
        nextHopGPU = rPSO_gpu.decodeRoutingGBestToNextHop();
        approxLifetimeGPU = rPSO_gpu.getApproxLifetime();
        });

    double speedup_r = cpuTime_r / gpuTime_r;
    csv << "Routing," << cpuTime_r << "," << gpuTime_r << "," << speedup_r << "\n";

    std::cout << "[Routing PSO] CPU: " << cpuTime_r << "s | GPU: " << gpuTime_r
        << "s | Speedup = " << speedup_r << "x\n";
#else
    csv << "Routing," << cpuTime_r << ",-,-\n";
    std::cout << "[Routing PSO] CPU: " << cpuTime_r << "s\n";
#endif

    // ======================================================
    // === CLUSTERIZAÇÃO (CPU e GPU) ========================
    // ======================================================
    std::cout << "\n========== CLUSTERIZAÇÃO ==========\n";

    std::vector<int> sensorAssignmentCPU;
    std::vector<int> sensorAssignmentGPU;
    std::vector<double> clusterRadiiCPU = computeClusterRadii(approxLifetimeCPU, 80.0);
#ifdef USE_CUDA
    std::vector<double> clusterRadiiGPU = computeClusterRadii(approxLifetimeGPU, 80.0);
#endif

    double cpuTime_c = measureTime([&]() {
        ClusteringPSO cPSO_cpu(net, nextHopCPU, clusterRadiiCPU, swarmSize, iterations);
        sensorAssignmentCPU = cPSO_cpu.optimizeClustering();
        });

#ifdef USE_CUDA
    double gpuTime_c = measureTime([&]() {
        ClusteringPSO_CUDA cPSO_gpu(net, nextHopGPU, clusterRadiiGPU, swarmSize, iterations);
        cPSO_gpu.run();
        std::vector<double> gbestVecGPU = cPSO_gpu.getGBestHost();
        sensorAssignmentGPU = decodeClusteringGBestToAssignment(gbestVecGPU, net, nextHopGPU, clusterRadiiGPU);
        });

    double speedup_c = cpuTime_c / gpuTime_c;
    csv << "Clustering," << cpuTime_c << "," << gpuTime_c << "," << speedup_c << "\n";
    std::cout << "[Clustering PSO] CPU: " << cpuTime_c << "s | GPU: " << gpuTime_c
        << "s | Speedup = " << speedup_c << "x\n";
#else
    csv << "Clustering," << cpuTime_c << ",-,-\n";
    std::cout << "[Clustering PSO] CPU: " << cpuTime_c << "s\n";
#endif

    csv.close();
    std::cout << "\n[Export] speedup_summary.csv salvo com sucesso!\n";

    // ======================================================
    // === SIMULAÇÃO CPU vs GPU =============================
    // ======================================================
    std::cout << "\n========== SIMULAÇÃO ==========\n";

    Network netCPU = net;
#ifdef USE_CUDA
    Network netGPU = net;
#endif

    std::cout << "\n[Simulação CPU]\n";
    Simulation simCPU(netCPU, nextHopCPU, sensorAssignmentCPU, 1.0);
    simCPU.run(1000);

#ifdef USE_CUDA
    std::cout << "\n[Simulação GPU]\n";
    Simulation simGPU(netGPU, nextHopGPU, sensorAssignmentGPU, 1.0);
    simGPU.run(1000);
#endif

    std::cout << "\n[Fim do pipeline completo]\n";
    return 0;
}
