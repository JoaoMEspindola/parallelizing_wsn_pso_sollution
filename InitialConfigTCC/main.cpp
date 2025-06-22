#include <iostream>
#include <string>
#include "network.hpp"
#include "routing_pso.hpp"
#include "cluster_radius.hpp"
#include "clustering_pso.hpp"
#include "simulation.hpp"
#include "export.cpp"

int main() {
    // === PARÂMETROS INICIAIS ===
    int numSensors = 300;
    int numGateways = 60;
    double areaWidth = 500.0, areaHeight = 500.0;

    // === GERAR REDE ===
    Network net(numSensors, numGateways, areaWidth, areaHeight);
    net.generate();
    exportNetworkAndLinksToCSV(net, "network_full.csv");

    std::cout << "Rede gerada com " << numSensors << " sensores e " << numGateways << " gateways.\n";

    // === PSO DE ROTEAMENTO ===
    RoutingPSO rPSO(net, 50, 100); // 50 partículas, 100 iterações
    std::vector<int> nextHop = rPSO.optimizeRouting();

    exportNextHops(nextHop, "nextHop.csv");

    std::vector<double> approxLifetime = rPSO.getApproxLifetime();

    std::cout << "\nRoteamento otimizado (nextHop):\n";
    for (int i = 0; i < nextHop.size(); ++i) {
        std::cout << "Gateway " << i << " -> "
            << (nextHop[i] == -1 ? "BS" : std::to_string(nextHop[i])) << "\n";
    }

    // === CÁLCULO DOS RAIO DE CLUSTER DESIGUAIS ===
    std::vector<double> clusterRadii = computeClusterRadii(approxLifetime, 80.0);

    // === PSO DE CLUSTERIZAÇÃO ===
    ClusteringPSO cPSO(net, nextHop, clusterRadii, 50, 100);
    std::vector<int> sensorAssignment = cPSO.optimizeClustering();

    std::cout << "\nAtribuição de sensores:\n";
    for (int i = 0; i < sensorAssignment.size(); ++i) {
        std::cout << "Sensor " << i << " -> Gateway " << sensorAssignment[i] << "\n";
    }

    // === SIMULAÇÃO DA REDE ===
    std::cout << "\nIniciando simulação...\n";
    Simulation sim(net, nextHop, sensorAssignment, 1.0); // Threshold_Energy = 1.0 J
    sim.run(1000); // até 1000 rodadas



    return 0;
}
