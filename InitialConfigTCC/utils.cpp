#include "utils.hpp"
#include "network.hpp"
#include <algorithm>
#include <cmath>

std::vector<int> decodeClusteringGBestToAssignment(
    const std::vector<double>& pos,
    const Network& net,
    const std::vector<int>& nextHop,
    const std::vector<double>& clusterRadii)
{
    int numSensors = net.numSensors;
    int numGateways = net.numGateways;
    std::vector<int> assignment(numSensors, -1);

    for (int i = 0; i < numSensors; ++i) {
        const Node& sensor = net.nodes[numGateways + i];
        std::vector<int> candidates;

        // encontra gateways elegíveis (ativos e dentro do raio)
        for (int g = 0; g < numGateways; ++g) {
            if (nextHop[g] == -2) continue;
            double dx = sensor.x - net.nodes[g].x;
            double dy = sensor.y - net.nodes[g].y;
            double d = std::sqrt(dx * dx + dy * dy);
            if (d <= clusterRadii[g]) candidates.push_back(g);
        }

        if (candidates.empty()) continue;
        double posVal = pos[i];
        int idx = static_cast<int>(posVal * candidates.size());
        if (idx >= static_cast<int>(candidates.size())) idx = static_cast<int>(candidates.size()) - 1;
        assignment[i] = candidates[idx];
    }

    return assignment;
}
