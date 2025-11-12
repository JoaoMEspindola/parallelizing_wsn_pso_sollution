#ifndef UTILS_HPP
#define UTILS_HPP
#pragma once

#include <vector>
#include <cmath>
#include <random>

// --- Forward declaration ---
class Network;

// --- Funções utilitárias básicas ---
inline double distance(double x1, double y1, double x2, double y2) {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

inline double randDouble(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

template<typename T>
T clamp(T v, T lo, T hi) {
    return std::max(lo, std::min(v, hi));
}

// --- Somente a declaração da função complexa ---
std::vector<int> decodeClusteringGBestToAssignment(
    const std::vector<double>& pos,
    const Network& net,
    const std::vector<int>& nextHop,
    const std::vector<double>& clusterRadii);

#endif // UTILS_HPP
