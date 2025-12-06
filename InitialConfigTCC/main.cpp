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
#include "routing_pso_block_cuda.hpp"  // <--- NOVA ESTRATÉGIA ADICIONADA
#include "clustering_pso_cuda_blocks.hpp"
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
    const int numSensors = 1200;
    const int numGateways = 120;
    const double areaWidth = 500.0, areaHeight = 500.0;
    const int swarmSize = 200;
    const int iterations = 400;

    int numBlocks_cluster = 8;
    int swarmPerBlock_cluster = 64;

    // === GERA REDE ===
    Network net(numSensors, numGateways, areaWidth, areaHeight);
    net.generate();
    
    std::cout << "Rede gerada com " << numSensors << " sensores e "
        << numGateways << " gateways.\n";

    std::ofstream csv("speedup_summary.csv");
    csv << "Stage,CPU_Time(s),GPU_ThreadPSO_Time(s),GPU_BlockPSO_Time(s),Speedup_ThreadPSO,Speedup_BlockPSO\n";

    // ======================================================
    // === ROTEAMENTO (CPU / GPU thread / GPU block) ========
    // ======================================================
    std::cout << "\n========== ROTEAMENTO ==========\n";

    std::vector<int> nextHopCPU, nextHopGPU_thread, nextHopGPU_block;
    std::vector<double> approxLifetimeCPU, approxLifetimeGPU_thread, approxLifetimeGPU_block;

    // ---- CPU PSO ----
    /*double cpuTime_r = measureTime([&]() {
        RoutingPSO rPSO_cpu(net, swarmSize, iterations);
        nextHopCPU = rPSO_cpu.optimizeRouting();
        approxLifetimeCPU = rPSO_cpu.getApproxLifetime();
        });*/

#ifdef USE_CUDA
    // ---- GPU PSO (thread-per-particle) ----
    double gpuTime_thread = measureTime([&]() {
        RoutingPSO_CUDA rPSO_gpu(net, swarmSize, iterations);
        rPSO_gpu.run();
        nextHopGPU_thread = rPSO_gpu.decodeRoutingGBestToNextHop();
        approxLifetimeGPU_thread = rPSO_gpu.getApproxLifetime();
        });

    // ---- GPU PSO (block-per-swarm) ----
    double gpuTime_block = measureTime([&]() {
        RoutingPSO_Block_CUDA rPSO_block(net, swarmSize, iterations, numBlocks_cluster); // 8 revoadas
        rPSO_block.run();
        nextHopGPU_block = rPSO_block.decodeRoutingGBestToNextHop();
        approxLifetimeGPU_block = rPSO_block.getGBestHost();
        });

    /*double speed_thread = cpuTime_r / gpuTime_thread;
    double speed_block = cpuTime_r / gpuTime_block;*/

    csv << "Routing," 
        //<< cpuTime_r
        << "," << gpuTime_thread << ","
        << gpuTime_block << ","
        //<< speed_thread << "," << speed_block
        << "\n";

    std::cout
        //<< "[Routing PSO] CPU: " << cpuTime_r
        << "s | GPU(Thread): " << gpuTime_thread
        << "s | GPU(Block): " << gpuTime_block
        //<< "s | Speedup(Thread)=" << speed_thread
        //<< "x | Speedup(Block)=" << speed_block
        << "x\n";
#else
    /*csv << "Routing," << cpuTime_r << ",-,-,-,-\n";
    std::cout << "[Routing PSO] CPU: " << cpuTime_r << "s\n";*/
#endif

    // ======================================================
    // === CLUSTERIZAÇÃO (CPU e GPU thread) =================
    // ======================================================
    std::cout << "\n========== CLUSTERIZAÇÃO ==========\n";

    std::vector<int> sensorAssignmentCPU, sensorAssignmentGPU;
    std::vector<int> sensorAssignmentGPU_block; // ADICIONADO: resultado do block
    //std::vector<double> clusterRadiiCPU = computeClusterRadii(approxLifetimeCPU, 80.0);

#ifdef USE_CUDA
    std::vector<double> clusterRadiiGPU = computeClusterRadii(approxLifetimeGPU_thread, 80.0);
    std::vector<double> clusterRadiiGPU_Block = computeClusterRadii(approxLifetimeGPU_block, 80.0);
#endif

    // ---- CPU PSO ----
    /*double cpuTime_c = measureTime([&]() {
        ClusteringPSO cPSO_cpu(net, nextHopCPU, clusterRadiiCPU, swarmSize, iterations);
        sensorAssignmentCPU = cPSO_cpu.optimizeClustering();
        });*/

#ifdef USE_CUDA
    // ---- GPU PSO (thread-per-particle) ----
    double gpuTime_c_thread = measureTime([&]() {
        ClusteringPSO_CUDA cPSO_gpu(net, nextHopGPU_thread, clusterRadiiGPU, swarmSize, iterations);
        cPSO_gpu.run();
        std::vector<double> gbestVecGPU = cPSO_gpu.getGBestHost();
        sensorAssignmentGPU = decodeClusteringGBestToAssignment(
            gbestVecGPU, net, nextHopGPU_thread, clusterRadiiGPU);
        });

    //double speedup_c_thread = cpuTime_c / gpuTime_c_thread;

    // ------------------------------------------------------
    // ---- GPU (BLOCK-PER-SWARM) CLUSTERING PSO -------------
    // ------------------------------------------------------
    
    if (swarmPerBlock_cluster < 1) {
        swarmPerBlock_cluster = 1;
        numBlocks_cluster = swarmSize;
    }

    double gpuTime_c_block = measureTime([&]() {
        ClusteringPSO_CUDA_Block cPSO_block(
            net,
            nextHopGPU_thread,
            clusterRadiiGPU,
            swarmPerBlock_cluster,
            numBlocks_cluster,
            iterations
        );
        cPSO_block.run();

        std::vector<double> gbestBlock = cPSO_block.getGBestHost();

        sensorAssignmentGPU_block = decodeClusteringGBestToAssignment(
            gbestBlock, net, nextHopGPU_block, clusterRadiiGPU_Block);
        });

    //double speedup_c_block = cpuTime_c / gpuTime_c_block;

    // ---- EXPORTAÇÃO E PRINTS ----
    csv << "Clustering,"
        //<< cpuTime_c << ","
        << gpuTime_c_thread << ","
        << gpuTime_c_block << ",";
        //<< speedup_c_thread << ","
        //<< speedup_c_block << "\n";

        std::cout
        //<< "[Clustering PSO] CPU: " << cpuTime_c << "s"
        << " | GPU(Thread): " << gpuTime_c_thread << "s"
        << " | GPU(Block): " << gpuTime_c_block << "s";
        //<< " | Speedup(Thread)=" << speedup_c_thread << "x"
        //<< " | Speedup(Block)=" << speedup_c_block << "x\n";

#else
    /*csv << "Clustering," << cpuTime_c << ",-,-,-,-\n";
    std::cout << "[Clustering PSO] CPU: " << cpuTime_c << "s\n";*/
#endif

    csv.close();
    // ======================================================
    // === SIMULAÇÃO =======================================
    // ======================================================
    std::cout << "\n========== SIMULAÇÃO ==========\n";

    //Network netCPU = net;
#ifdef USE_CUDA
    Network netGPU = net;
    Network netGPU_block = net;
#endif

    // ---- Simulação CPU ----
    /*std::cout << "\n[Simulação CPU]\n";
    Simulation simCPU(netCPU, nextHopCPU, sensorAssignmentCPU, 1.0);
    simCPU.run(1000);*/

#ifdef USE_CUDA
    // ---- Simulação GPU(Thread) ----
    std::cout << "\n[Simulação GPU(Thread)]\n";
    Simulation simGPU_thread(netGPU, nextHopGPU_thread, sensorAssignmentGPU, 1.0);
    simGPU_thread.run(1000);

    // ---- Simulação GPU(Block) ----
    std::cout << "\n[Simulação GPU(Block)]\n";
    Simulation simGPU_block(netGPU_block, nextHopGPU_block, sensorAssignmentGPU_block, 1.0);
    simGPU_block.run(1000);
#endif

    std::cout << "\n[Fim do pipeline completo]\n";
    return 0;
}
