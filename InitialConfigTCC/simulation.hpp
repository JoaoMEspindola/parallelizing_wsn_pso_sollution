#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "network.hpp"
#include "energy.hpp"
#include <iostream>
#include <unordered_set>

class Simulation {
public:
    Simulation(Network& net_,
        std::vector<int> nextHop_,
        std::vector<int> sensorAssignment_,
        double thresholdEnergy_ = 1.0)
        : net(net_), nextHop(nextHop_), sensorAssignment(sensorAssignment_),
        thresholdEnergy(thresholdEnergy_) {
    }

    void run(int maxRounds = 1000) { 
        int round = 0;

		while (round < maxRounds && activeGateways().size() > 0) { // Enquanto houver gateways ativos e não atingir o máximo de rodadas
            std::cout << "\n--- Rodada " << round << " ---\n";
            double minE = 1e9;
            for (int g = 0; g < net.numGateways; ++g)
                if (isActive(g))
                    minE = std::min(minE, net.nodes[g].energy);

            std::cout << "Energia mínima entre gateways ativos: " << minE << "\n";

            // === 1. SENSORES enviam para seus CHs ===
            for (int i = 0; i < net.numSensors; ++i) {
                int sid = net.numGateways + i;
                int gid = sensorAssignment[i];

                if (!isValidGateway(gid)) continue;

                Node& sensor = net.nodes[sid];
                Node& gateway = net.nodes[gid];

                double d = distance(sensor.x, sensor.y, gateway.x, gateway.y);
                sensor.energy -= transmitEnergy(d);

                gateway.energy -= receiveEnergy();                  // recebe pacote
                gateway.energy -= 5e-9 * PACKET_SIZE;               // agregação (E_DA)
            }

            // === 2. GATEWAYS transmitem para seu next hop ===
            for (int g = 0; g < net.numGateways; ++g) {
                if (!isActive(g)) continue;
                int nh = nextHop[g];

                if (nh == -1) {
                    // Direto para BS
                    double d = distance(net.nodes[g].x, net.nodes[g].y, net.bs.x, net.bs.y);
                    net.nodes[g].energy -= transmitEnergy(d);
                }
                else if (isActive(nh)) {
                    double d = distance(net.nodes[g].x, net.nodes[g].y, net.nodes[nh].x, net.nodes[nh].y);
                    net.nodes[g].energy -= transmitEnergy(d);
                    net.nodes[nh].energy -= receiveEnergy();
                }
            }

            // === 3. Detectar falhas de CHs (Threshold) ===
            for (int g = 0; g < net.numGateways; ++g) {
                if (isActive(g) && net.nodes[g].energy < thresholdEnergy) {
                    std::cout << "⚠️  Gateway " << g << " falhou (energia abaixo de limiar).\n";
                    deadGateways.insert(g);

                    if (!firstCHFailed) {
                        firstCHFailureRound = round;
                        firstCHFailed = true;
                        std::cout << "⛔ Primeira falha de CH detectada na rodada " << round << ".\n";
                    }
                }
            }

            // === 4. SENSORES órfãos enviam HELP (e buscam novo CH) ===
            for (int i = 0; i < net.numSensors; ++i) {
                int sid = net.numGateways + i;
                int gid = sensorAssignment[i];

                if (!isValidGateway(gid)) {
                    // Sensor envia HELP
                    int newCH = findNewGateway(sid);
                    if (newCH != -1) {
                        sensorAssignment[i] = newCH;
                        std::cout << "Sensor " << i << " se reconectou ao Gateway " << newCH << " após HELP.\n";
                    }
                    else {
                        // continua órfão
                        sensorAssignment[i] = -1;
                    }
                }
            }

            ++round;
            std::cout << "Energia do Gateway 0: " << net.nodes[0].energy << "\n";
            std::cout << "Energia do Sensor 0: " << net.nodes[net.numGateways].energy << "\n";

        }

        std::cout << "\n=== Fim da simulação ===\n";
        std::cout << "Rodadas executadas: " << round << "\n";
        std::cout << "Gateways ativos restantes: " << activeGateways().size() << "\n";

        if (firstCHFailureRound >= 0) {
            std::cout << "🕒 Primeira falha de gateway ocorreu na rodada " << firstCHFailureRound << ".\n";
        }
        else {
            std::cout << "✅ Nenhum CH falhou durante a simulação.\n";
        }

    }


private:
    Network& net;
    std::vector<int> nextHop;
    std::vector<int> sensorAssignment;
    std::unordered_set<int> deadGateways;
    double thresholdEnergy;
    int firstCHFailureRound = -1;
    bool firstCHFailed = false;

    bool isActive(int gid) const {
        return deadGateways.find(gid) == deadGateways.end();
    }

    bool isValidGateway(int gid) const {
        return gid >= 0 && gid < net.numGateways && isActive(gid) && nextHop[gid] != -2;
    }

    int findNewGateway(int sensorId) {
        const Node& s = net.nodes[sensorId];
        int best = -1;
        double minDist = std::numeric_limits<double>::max();

        for (int g = 0; g < net.numGateways; ++g) {
            if (!isValidGateway(g)) continue;

            double d = distance(s.x, s.y, net.nodes[g].x, net.nodes[g].y);
            if (d <= 80.0 && d < minDist) {
                minDist = d;
                best = g;
            }
        }

        return best;
    }

    std::vector<int> activeGateways() const {
        std::vector<int> active;
        for (int g = 0; g < net.numGateways; ++g)
            if (isActive(g))
                active.push_back(g);
        return active;
    }


};

#endif
