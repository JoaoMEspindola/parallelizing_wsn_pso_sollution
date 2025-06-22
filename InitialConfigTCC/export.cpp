#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <algorithm>
#include "network.hpp"

// Exporta nós e links num único CSV com 9 colunas:
// recordType,id,x,y,isGateway,energy,from,to,edgeType
void exportNetworkAndLinksToCSV(const Network& net, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir o arquivo para exportação: " << filename << "\n";
        return;
    }

    int numNodes = net.nodes.size();
    int bsId = numNodes;  // BS recebe o próximo ID inteiro

    // Cabeçalho
    file << "recordType,id,x,y,isGateway,energy,from,to,edgeType\n";

    // Exportar os nós (sensores e gateways)
    for (const auto& node : net.nodes) {
        file << "node," << node.id << "," << node.x << "," << node.y << ","
            << (node.isGateway ? 1 : 0) << "," << node.energy << ",,,\n";
    }

    // Exportar a BS como nó
    file << "node," << bsId << "," << net.bs.x << "," << net.bs.y << ",1," << 1e9 << ",,,\n";

    // Exportar as arestas (links) - apenas Sensor→Gateway, Gateway→Gateway e Gateway→BS
    std::set<std::pair<int, int>> uniqueLinks;

    for (const auto& node : net.nodes) {
        for (int neighborId : node.neighbors) {
            int a = std::min(node.id, neighborId);
            int b = std::max(node.id, neighborId);

            if (uniqueLinks.insert({ a, b }).second) {
                bool isG_a = net.nodes[a].isGateway;
                bool isG_b = net.nodes[b].isGateway;

                // Ignorar sensor-sensor
                if (!isG_a && !isG_b) continue;

                std::string linkType;
                if (!isG_a && isG_b) linkType = "Sensor-Gateway";
                else if (isG_a && !isG_b) linkType = "Gateway-Sensor";
                else if (isG_a && isG_b) linkType = "Gateway-Gateway";

                file << "edge,,,,," << a << "," << b << "," << linkType << "\n";
            }
        }
    }

    // Exportar conexões entre gateways e a BS (se dentro do alcance)
    for (int g = 0; g < net.numGateways; ++g) {
        const Node& gw = net.nodes[g];
        double d = distance(gw.x, gw.y, net.bs.x, net.bs.y);

        if (d <= net.gatewayRange) {
            int a = std::min(g, bsId);
            int b = std::max(g, bsId);

            if (uniqueLinks.insert({ a, b }).second) {
                file << "edge,,,,," << a << "," << b << "," << "Gateway-BS" << "\n";
            }
        }
    }

    file.close();
    std::cout << "Exportação concluída: " << filename << "\n";
}

void exportNextHops(const std::vector<int>& nextHop, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir arquivo para exportar nextHop: " << filename << "\n";
        return;
    }

    file << "gateway,nextHop\n";
    for (int g = 0; g < nextHop.size(); ++g) {
        if (nextHop[g] >= -1) {  // Exporta -1 (BS) e >=0 (outros gateways)
            file << g << "," << nextHop[g] << "\n";
        }
    }
    file.close();
    std::cout << "Exportação de nextHop concluída em: " << filename << "\n";
}