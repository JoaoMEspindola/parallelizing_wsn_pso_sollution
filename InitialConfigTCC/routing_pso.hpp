#ifndef ROUTING_PSO_HPP
#define ROUTING_PSO_HPP

#include "network.hpp"
#include "utils.hpp"
#include "energy.hpp"
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> pbest;
    double pbestFitness = -1;
};

class RoutingPSO {
public:
    RoutingPSO(Network& network_, int swarmSize_, int iterations_)
        : network(network_), swarmSize(swarmSize_), iterations(iterations_) {
        numGateways = network.numGateways;
    }

    const std::vector<double>& getApproxLifetime() const {
        return lastApproxLifetime;
    }

    // Result: nextHop[i] = ID do próximo gateway ou -1 para a BS
    std::vector<int> optimizeRouting() {
        initializeParticles();

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
                }
            }

            updateParticles();
        }

        return decodeParticle(globalBest);
    }


private:
    Network& network;
    int numGateways;
    int swarmSize;
    int iterations;

    std::vector<Particle> swarm;
    std::vector<double> globalBest;
    std::vector<double> lastApproxLifetime;

    double globalBestFitness = -1;

	void initializeParticles() { // inicializa o swarm com partículas aleatórias
        swarm.clear();
        for (int i = 0; i < swarmSize; ++i) {
            Particle p;
            p.position.resize(numGateways);
            p.velocity.resize(numGateways);
            p.pbest.resize(numGateways);
            for (int d = 0; d < numGateways; ++d) {
                p.position[d] = randDouble(0.0, 1.0);
                p.velocity[d] = randDouble(-0.5, 0.5);
                p.pbest[d] = p.position[d];
            }
            swarm.push_back(p);
        }
    }

	std::vector<int> decodeParticle(const std::vector<double>& pos) { // converte a posição da partícula em próximos saltos
        std::vector<int> result(numGateways, -1);
        for (int i = 0; i < numGateways; ++i) {
            std::vector<int> candidates = network.getNextHopCandidates(i);
            if (candidates.empty()) continue;
            int idx = std::min((int)(pos[i] * candidates.size()), (int)candidates.size() - 1);
            result[i] = candidates[idx];
        }
        return result;
    }

    double evaluate(const Particle& p) { // avalia a partícula e retorna a fitness (vida útil aproximada)
  //      if (p.position.empty()) return 0.0; // Partícula vazia
  //      if (std::any_of(p.position.begin(), p.position.end(), [](double v) { return v < 0.0 || v > 1.0; })) {
  //          return 0.0; // Posição inválida
  //      }
  //      if (std::all_of(p.position.begin(), p.position.end(), [](double v) { return v == 0.0; })) {
  //          return 0.0; // Todos os gateways com probabilidade zero
		//}
		auto nextHop = decodeParticle(p.position); //pode ser paralelizado (1 thread por gateway)
        std::vector<double> approxLifetime(numGateways);

        for (int i = 0; i < numGateways; ++i) {
            const Node& g = network.nodes[i];
            int nh = nextHop[i];

            if (nh == -1) {
                // Comunicação direta com a BS
                double d = distance(g.x, g.y, network.bs.x, network.bs.y);
                double e_tx = transmitEnergy(d);
                approxLifetime[i] = g.energy / e_tx;
            }
            else {
                const Node& next = network.nodes[nh];
                double d = distance(g.x, g.y, next.x, next.y);
                double e_tx = transmitEnergy(d);
                double e_rx = receiveEnergy(); // Considera receber de outros nós
                int received = countReceived(i, nextHop);
                double total = received * e_rx + (received + 1) * e_tx;
                approxLifetime[i] = g.energy / total;
            }
        }

        lastApproxLifetime = approxLifetime;

        return *std::min_element(approxLifetime.begin(), approxLifetime.end()); //warp-level reduction ??
    }

	int countReceived(int id, const std::vector<int>& nextHop) { // conta quantos nós estão enviando para o gateway id
        int count = 0;
        for (int i = 0; i < nextHop.size(); ++i)
            if (nextHop[i] == id) ++count;
        return count;
    }

	void updateParticles() { // atualiza a posição e velocidade das partículas
        const double w = 0.7968;
        const double c1 = 1.4962;
        const double c2 = 1.4962;

        for (auto& p : swarm) {
            for (int d = 0; d < numGateways; ++d) {
                double r1 = randDouble(0.0, 1.0);
                double r2 = randDouble(0.0, 1.0);

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
