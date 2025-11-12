#ifndef COMPACT_GRAPH_HPP
#define COMPACT_GRAPH_HPP

#include "network.hpp"
#include <vector>

// Estrutura simplificada do nó para GPU
struct NodeGPU {
    double x, y;     // posição
    double energy;   // energia residual
    int isGateway;   // 1 = gateway, 0 = sensor
};

// Estrutura da lista compacta (CPU)
struct CompactGraphHost {
    std::vector<NodeGPU> h_nodes;
    std::vector<int> h_offsets;
    std::vector<int> h_adjacency;
};

// === Função para gerar a lista compacta ===
inline CompactGraphHost buildCompactGraph(const Network& net) {
    CompactGraphHost g;
    int N = net.nodes.size();

    g.h_nodes.resize(N);
    g.h_offsets.resize(N + 1);

    int totalEdges = 0;
    for (int i = 0; i < N; ++i)
        totalEdges += net.nodes[i].neighbors.size();

    g.h_adjacency.reserve(totalEdges);

    int offset = 0;
    for (int i = 0; i < N; ++i) {
        g.h_offsets[i] = offset;

        const Node& n = net.nodes[i];
        NodeGPU gpuNode = { n.x, n.y, n.energy, n.isGateway ? 1 : 0 };
        g.h_nodes[i] = gpuNode;

        for (int nb : n.neighbors)
            g.h_adjacency.push_back(nb);

        offset += n.neighbors.size();
    }

    g.h_offsets[N] = offset;
    return g;
}

#endif
