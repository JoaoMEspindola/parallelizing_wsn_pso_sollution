#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <algorithm>
#include "network.hpp"

// Exporta nós e links num único CSV com 9 colunas:
// recordType,id,x,y,isGateway,energy,from,to,edgeType
void exportNetworkAndLinksToCSV(
    const Network& net,
    const std::string& filename,
    const std::vector<int>& nextHop,
    const std::vector<int>& clusterAssignment,
    const std::vector<double>& clusterRadii);