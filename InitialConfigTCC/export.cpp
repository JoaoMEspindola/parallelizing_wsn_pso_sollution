#include "export.hpp"

void exportNetworkAndLinksToCSV(
    const Network& net,
    const std::string& filename,
    const std::vector<int>& nextHop,
    const std::vector<int>& clusterAssignment,
    const std::vector<double>& clusterRadii)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro ao abrir arquivo: " << filename << "\n";
        return;
    }

    int numNodes = net.nodes.size();
    int bsId = numNodes;

    file << "recordType,id,x,y,isGateway,energy,from,to,edgeType,nextHop,cluster,clusterRadius\n";

    // ---- export nodes ----
    for (const auto& node : net.nodes) {
        int id = node.id;
        bool gw = node.isGateway;

        file << "node,"
            << id << ","
            << node.x << ","
            << node.y << ","
            << (gw ? 1 : 0) << ","
            << node.energy << ",,,,";

        // nextHop: só gateways possuem
        if (gw && id < (int)nextHop.size())
            file << nextHop[id];
        else
            file << -2;

        file << ",";

        // cluster assignment: só sensores possuem
        if (!gw && (id - net.numGateways) < (int)clusterAssignment.size())
            file << clusterAssignment[id - net.numGateways];
        else
            file << -1;

        file << ",";

        // cluster radius: só gateways possuem
        if (gw && id < (int)clusterRadii.size())
            file << clusterRadii[id];
        else
            file << -1.0;

        file << "\n";
    }

    // ---- export BS ----
    file << "node," << bsId << "," << net.bs.x << "," << net.bs.y
        << ",1," << 1e9 << ",,,,-1,-1,-1\n";

    // ---- export unique edges ----
    std::set<std::pair<int, int>> uniqueEdges;

    // ------------------------------------------------------
// EDGES SENSOR↔GATEWAY / GATEWAY↔GATEWAY
// ------------------------------------------------------
    for (const auto& node : net.nodes) {
        for (int nid : node.neighbors) {

            int a = std::min(node.id, nid);
            int b = std::max(node.id, nid);

            if (!uniqueEdges.insert({ a,b }).second)
                continue;

            // classify edge
            bool isA_gw = net.nodes[a].isGateway;
            bool isB_gw = net.nodes[b].isGateway;

            std::string etype;
            if (!isA_gw && isB_gw) etype = "Sensor-Gateway";
            else if (isA_gw && !isB_gw) etype = "Gateway-Sensor";
            else etype = "Gateway-Gateway";

            // 12 COLUNAS FIXAS
            file
                << "edge"  // recordType
                << ",,"    // id
                << ",,"    // x,y
                << ",,"    // isGateway,energy
                << a << "," << b << ","   // from,to
                << etype << ","           // edgeType
                << "-1,-1,-1\n";          // nextHop,cluster,clusterRadius
        }
    }


    // edges Gateway→BS
    for (int g = 0; g < net.numGateways; ++g) {
        const auto& gw = net.nodes[g];
        double d = distance(gw.x, gw.y, net.bs.x, net.bs.y);

        if (d <= net.gatewayRange) {
            int a = std::min(g, bsId);
            int b = std::max(g, bsId);

            if (uniqueEdges.insert({ a,b }).second) {

                file
                    << "edge"  // recordType
                    << ",,"    // id
                    << ",,"    // x,y
                    << ",,"    // isGateway,energy
                    << a << "," << b << ","   // from,to
                    << "Gateway-BS,"          // edgeType
                    << "-1,-1,-1\n";          // nextHop,cluster,clusterRadius
            }
        }
    }

    file.close();
    std::cout << "[EXPORT] CSV gerado: " << filename << "\n";
}
