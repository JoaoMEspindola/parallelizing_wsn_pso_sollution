#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "node.hpp"
#include "utils.hpp"
#include <iostream>

struct BaseStation {
    double x = 500.0;
    double y = 250.0;
};

class Network {
public:
    std::vector<Node> nodes;
    int numSensors, numGateways;
    double width, height;
    double sensorRange, gatewayRange;
    BaseStation bs;

    Network(int numSensors_, int numGateways_, double width_, double height_,
        double sensorRange_ = 80.0, double gatewayRange_ = 120.0)
        : numSensors(numSensors_), numGateways(numGateways_),
        width(width_), height(height_), sensorRange(sensorRange_), gatewayRange(gatewayRange_) {
    }

	void generate() { // Gera os nós da rede
        int id = 0;
        // Gateways
        for (int i = 0; i < numGateways; ++i)
            nodes.emplace_back(id++, randDouble(0, width), randDouble(0, height), true, 10.0);

        // Sensor nodes
        for (int i = 0; i < numSensors; ++i)
            nodes.emplace_back(id++, randDouble(0, width), randDouble(0, height), false, 2.0);

        computeNeighbors();
    }

    void computeNeighbors() {
        for (int i = 0; i < nodes.size(); ++i) {
            for (int j = 0; j < nodes.size(); ++j) {
                if (i == j) continue;

                double d = distance(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);

                // Gateway → Gateway
                if (nodes[i].isGateway && nodes[j].isGateway && d <= gatewayRange) {
                    nodes[i].neighbors.push_back(j);
                }
                // Sensor → Gateway
                else if (!nodes[i].isGateway && nodes[j].isGateway && d <= sensorRange) {
                    nodes[i].neighbors.push_back(j);
                }
                // Nenhuma outra conexão é permitida pelo artigo
            }
        }
    }

    std::vector<int> getNextHopCandidates(int gatewayId) const {
        const Node& g = nodes[gatewayId];
        std::vector<int> candidates;
        double dist_i_bs = distance(g.x, g.y, bs.x, bs.y);

        // 1) Gateway → Gateway: só outros gateways dentro de gatewayRange e com distância até BS <= do próprio
        for (int j = 0; j < numGateways; ++j) {
            if (j == gatewayId) continue;
            const Node& neighbor = nodes[j];
            double dij = distance(g.x, g.y, neighbor.x, neighbor.y);
            double dist_j_bs = distance(neighbor.x, neighbor.y, bs.x, bs.y);
            if (dij <= gatewayRange && dist_j_bs <= dist_i_bs) {
                candidates.push_back(j);
            }
        }
        // 2) Gateway → BS direto: somente se dentro de gatewayRange
        if (dist_i_bs <= gatewayRange) {
            candidates.push_back(-1); // -1 indica BS
        }
        return candidates;
    }


    void printSummary() {
        std::cout << "Total Nodes: " << nodes.size() << "\n";
        for (auto& node : nodes) {
            std::cout << "Node " << node.id
                << (node.isGateway ? " [Gateway]" : " [Sensor]")
                << " (" << node.x << ", " << node.y << ") -> "
                << node.neighbors.size() << " neighbors\n";
        }
    }
};

#endif
