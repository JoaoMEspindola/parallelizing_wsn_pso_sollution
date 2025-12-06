#ifndef CLUSTERING_PSO_HPP
#define CLUSTERING_PSO_HPP

#include "export.hpp"
#include "energy.hpp"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <chrono>

struct ClusteringParticle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> pbest;
    double pbestFitness = -1;
};

class ClusteringPSO {
public:
    ClusteringPSO(Network& network_,
        const std::vector<int>& nextHop_,
        const std::vector<double>& clusterRadii_,
        int swarmSize_, int iterations_)
        : network(network_), nextHop(nextHop_), clusterRadii(clusterRadii_),
        swarmSize(swarmSize_), iterations(iterations_) {
        numSensors = network.numSensors;
        numGateways = network.numGateways;
    }

	std::vector<int> optimizeClustering() { // otimização do agrupamento de sensores
        initializeParticles();
        std::vector<double> bestHistory;
        bestHistory.reserve(iterations);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < iterations; ++t) {
            for (auto& p : swarm) {
                double fitness = evaluate(p);
                if (fitness > p.pbestFitness) {
                    p.pbestFitness = fitness;
                    p.pbest = p.position;
                }

                if (fitness > globalBestFitness) {
                    globalBestFitness = fitness;
                    globalBest = p.position;

                    auto tNow = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(tNow - t0).count();
                    gbestTimeline.emplace_back(ms, globalBestFitness);
                }
            }
            bestHistory.push_back(globalBestFitness);
            updateParticles();
        }
        // depois de salvar timeline normal
        auto tEnd = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(tEnd - t0).count();
        std::ofstream csv("pso_convergence_cpu_clustering.csv");
        csv << "iteration,best_fitness\n";
        for (int i = 0; i < (int)bestHistory.size(); ++i)
            csv << i << "," << bestHistory[i] << "\n";

        std::ofstream out("gbest_timeline_cpu.csv");
        out << "time_ms,fitness\n";
        for (auto& p : gbestTimeline)
            out << p.first << "," << p.second << "\n";
        out << "END," << total_ms << "\n";
        out.close();


        std::vector<int> assignmentCPU = decodeParticle(globalBest);

        exportNetworkAndLinksToCSV(
            network,
            "cpu_network.csv",
            nextHop,
            assignmentCPU,
            clusterRadii
        );


        return decodeParticle(globalBest);
    }

private:
    Network& network;
    const std::vector<int>& nextHop;
    const std::vector<double>& clusterRadii;
    int numSensors, numGateways;
    int swarmSize, iterations;

    std::vector<ClusteringParticle> swarm;
    std::vector<double> globalBest;
    double globalBestFitness = -1;
    std::vector<std::pair<double, double>> gbestTimeline;

	void initializeParticles() { // inicializa o enxame de partículas
        swarm.clear();
        for (int i = 0; i < swarmSize; ++i) {
            ClusteringParticle p;
            p.position.resize(numSensors);
            p.velocity.resize(numSensors);
            p.pbest.resize(numSensors);
            for (int d = 0; d < numSensors; ++d) {
                p.position[d] = randDouble(0.0, 1.0);
                p.velocity[d] = randDouble(-0.5, 0.5);
                p.pbest[d] = p.position[d];
            }
            swarm.push_back(p);
        }
    }

	std::vector<int> decodeParticle(const std::vector<double>& pos) { // decodifica a posição da partícula para atribuição de gateways aos sensores
        std::vector<int> assignment(numSensors, -1); 
        for (int i = 0; i < numSensors; ++i) {
            const Node& s = network.nodes[network.numGateways + i];

            // Gateways candidatos dentro do raio
            std::vector<int> candidates;
            for (int g = 0; g < numGateways; ++g) {
                const Node& gateway = network.nodes[g];
                if (nextHop[g] == -2) continue; // sem caminho válido
                double d = distance(s.x, s.y, gateway.x, gateway.y);
                if (d <= clusterRadii[g]) {
                    candidates.push_back(g);
                }
            }

            if (candidates.empty()) continue;
            int idx = std::min((int)(pos[i] * candidates.size()), (int)candidates.size() - 1);
            assignment[i] = candidates[idx];
        }
        return assignment;
    }

	double evaluate(const ClusteringParticle& p) { // avalia a partícula com base na energia de vida útil dos gateways
        auto assigned = decodeParticle(p.position); // 1 thread por sensor → decodifica localmente a escolha do gateway.
        std::vector<int> clusterSizes(numGateways, 0);
        for (int s = 0; s < numSensors; ++s) { // pode ser feito com atomicAdd() em um kernel paralelo por sensor.
            int g = assigned[s];
            if (g != -1) ++clusterSizes[g];
        }

        std::vector<double> lifetime(numGateways);
		for (int g = 0; g < numGateways; ++g) { // 1thread por gateway → calcula energia de cada gateway localmente.
            const Node& gw = network.nodes[g];

            double e_rx = receiveEnergy();
            double e_agg = 5e-9 * PACKET_SIZE; // 5 nJ/bit (agregação)
            double e_intra = clusterSizes[g] * (e_rx + e_agg);

            double e_inter;
            if (nextHop[g] == -1) {
                double d = distance(gw.x, gw.y, network.bs.x, network.bs.y);
                e_inter = transmitEnergy(d);
            }
            else if (nextHop[g] >= 0) {
                const Node& next = network.nodes[nextHop[g]];
                double d = distance(gw.x, gw.y, next.x, next.y);
                int relays = countRelaysTo(g); // quantos gateways enviam dados por ele / pode ser uma redução com atomic counts.
                e_inter = relays * e_rx + (relays + 1) * transmitEnergy(d);
            }
            else {
                e_inter = std::numeric_limits<double>::max();
            }

            double total = e_inter + e_intra;
            lifetime[g] = (total > 0) ? gw.energy / total : 0;
        }

        return *std::min_element(lifetime.begin(), lifetime.end());// redução entre lifetime[g] para encontrar o mínimo.
    }

	int countRelaysTo(int gId) { // conta quantos gateways enviam dados para o gateway gId
        int count = 0;
        for (int i = 0; i < nextHop.size(); ++i)
            if (nextHop[i] == gId) ++count;
        return count;
    }

	void updateParticles() { // atualiza a posição e velocidade das partículas
        const double w = 0.7968, c1 = 1.4962, c2 = 1.4962;
        for (auto& p : swarm) {
            for (int d = 0; d < numSensors; ++d) {
                double r1 = randDouble(0.0, 1.0), r2 = randDouble(0.0, 1.0);
                p.velocity[d] = w * p.velocity[d]
                    + c1 * r1 * (p.pbest[d] - p.position[d])
                    + c2 * r2 * (globalBest[d] - p.position[d]);
                p.velocity[d] = clamp(p.velocity[d], -0.5, 0.5);
                p.position[d] += p.velocity[d];
                p.position[d] = clamp(p.position[d], 0.0, 1.0);
            }
        }
    }
};

#endif
