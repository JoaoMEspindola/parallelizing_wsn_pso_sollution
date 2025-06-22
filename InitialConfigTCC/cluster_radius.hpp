#ifndef CLUSTER_RADIUS_HPP
#define CLUSTER_RADIUS_HPP

#include <vector>
#include <algorithm>

inline std::vector<double> computeClusterRadii( // Função para calcular os raios dos clusters
    const std::vector<double>& approxLifetime,
    double d_max,
    double epsilon = 0.5)
{
    std::vector<double> clusterRadii(approxLifetime.size());

    double minLife = *std::min_element(approxLifetime.begin(), approxLifetime.end());
    double maxLife = *std::max_element(approxLifetime.begin(), approxLifetime.end());

    for (size_t i = 0; i < approxLifetime.size(); ++i) {
        if (maxLife == minLife) {
            // Vida útil uniforme: todos têm mesmo raio
            clusterRadii[i] = d_max;
        }
        else {
            double factor = (maxLife - approxLifetime[i]) / (maxLife - minLife);
            clusterRadii[i] = (1.0 - epsilon * factor) * d_max;
        }
    }

    return clusterRadii;
}

#endif
