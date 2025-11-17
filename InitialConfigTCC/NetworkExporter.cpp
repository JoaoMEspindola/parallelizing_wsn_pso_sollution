#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include "network.hpp"   // define Network, Node
using std::ofstream;
using std::string;
using std::endl;

void exportNetworkSnapshot(
    const Network& net,
    const std::vector<double>& bestPos,        // tamanho numSensors
    const std::vector<int>& sensorOffsets,     // host
    const std::vector<int>& sensorAdj,         // host flattened
    const std::vector<double>& clusterRadii,   // gateway radii
    const std::string& filename)
{
    int numGateways = net.numGateways;
    int numSensors = net.numSensors;

    ofstream out(filename);
    out << std::fixed << std::setprecision(6);

    out << "{\n";
    out << "  \"numGateways\": " << numGateways << ",\n";
    out << "  \"numSensors\": " << numSensors << ",\n";

    out << "  \"baseStation\": {\"x\": " << net.bs.x << ", \"y\": " << net.bs.y << "},\n";

    // -------------------------
    // Gateways
    // -------------------------
    out << "  \"gateways\": [\n";
    for (int g = 0; g < numGateways; ++g) {
        const Node& n = net.nodes[g];
        out << "    {\"id\": " << g
            << ", \"x\": " << n.x
            << ", \"y\": " << n.y
            << ", \"radius\": " << clusterRadii[g]
            << ", \"energy\": " << n.energy
            << ", \"nextHop\": " << n.nextHop   // se não existir nextHop em Node, ignore
            << "}";
        if (g < numGateways - 1) out << ",";
        out << "\n";
    }
    out << "  ],\n";

    // -------------------------
    // Sensors + cluster assignment
    // -------------------------
    out << "  \"sensors\": [\n";
    for (int s = 0; s < numSensors; ++s) {

        int start = sensorOffsets[s];
        int end = sensorOffsets[s + 1];
        int deg = end - start;

        int assigned = -1;
        if (deg > 0) {
            int pick = (int)(bestPos[s] * deg);
            if (pick >= deg) pick = deg - 1;
            assigned = sensorAdj[start + pick];
        }

        const Node& n = net.nodes[numGateways + s];
        out << "    {\"id\": " << s
            << ", \"x\": " << n.x
            << ", \"y\": " << n.y
            << ", \"cluster\": " << assigned
            << "}";
        if (s < numSensors - 1) out << ",";
        out << "\n";
    }
    out << "  ]\n";

    out << "}\n";
    out.close();
}
