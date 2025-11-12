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

    void run(int maxRounds = 1000, bool stopOnFirstCHFailure = true, const std::string& csvOut = "simulation_trace.csv") {
        int round = 0;

        // states
        std::unordered_set<int> deadSensors; // sensor ids (sid)
        deadSensors.reserve(net.numSensors);
        deadGateways.clear(); // já existe na classe

        // prepare CSV
        std::ofstream csv(csvOut);
        csv << "round,alive_sensors,alive_gateways,avg_energy_sensors,avg_energy_gateways,firstCHFailureRound\n";

        // helper lambdas
        auto isSensorAlive = [&](int sid)->bool {
            return deadSensors.find(sid) == deadSensors.end() && net.nodes[sid].energy > 0.0;
            };
        auto countAliveSensors = [&]() -> int {
            int cnt = 0;
            for (int i = 0; i < net.numSensors; ++i) {
                int sid = net.numGateways + i;
                if (isSensorAlive(sid)) ++cnt;
            }
            return cnt;
            };
        auto avgEnergySensors = [&]() -> double {
            double sum = 0.0; int cnt = 0;
            for (int i = 0; i < net.numSensors; ++i) {
                int sid = net.numGateways + i;
                if (isSensorAlive(sid)) { sum += net.nodes[sid].energy; ++cnt; }
            }
            return (cnt > 0) ? (sum / cnt) : 0.0;
            };
        auto avgEnergyGateways = [&]() -> double {
            double sum = 0.0; int cnt = 0;
            for (int g = 0; g < net.numGateways; ++g) {
                if (isActive(g)) { sum += net.nodes[g].energy; ++cnt; }
            }
            return (cnt > 0) ? (sum / cnt) : 0.0;
            };

        // main loop
        while (round < maxRounds) {
            // stop conditions
            int aliveGws = (int)activeGateways().size();
            int aliveSensorsCount = countAliveSensors();
            if (aliveGws == 0 || aliveSensorsCount == 0) break;
            if (stopOnFirstCHFailure && firstCHFailed) break;

            // --- 1. Sensores transmitem para seus CHs (se estiverem vivos e tiverem CH válido) ---
            for (int i = 0; i < net.numSensors; ++i) {
                int sid = net.numGateways + i;
                if (!isSensorAlive(sid)) continue; // sensor morto, pula

                int gid = sensorAssignment[i];
                if (!isValidGateway(gid)) {
                    // órfão: não transmite (até tentar HELP na etapa 4)
                    continue;
                }

                Node& sensor = net.nodes[sid];
                Node& gateway = net.nodes[gid];

                // sensor transmite
                double d = distance(sensor.x, sensor.y, gateway.x, gateway.y);
                double e_tx = transmitEnergy(d);
                sensor.energy -= e_tx;
                if (sensor.energy <= 0.0) {
                    deadSensors.insert(sid);
                    // sensor morreu ao transmitir; continue (gateway may still have received)
                }

                // gateway recebe only if it's still active
                if (isActive(gid)) {
                    gateway.energy -= receiveEnergy();
                    gateway.energy -= 5e-9 * PACKET_SIZE; // E_DA
                    if (gateway.energy <= 0.0 && deadGateways.find(gid) == deadGateways.end()) {
                        deadGateways.insert(gid);
                        if (!firstCHFailed) { firstCHFailed = true; firstCHFailureRound = round; }
                    }
                }
            }

            // --- 2. Gateways transmitem para seu next hop (se estiverem ativos e vivos) ---
            for (int g = 0; g < net.numGateways; ++g) {
                if (!isActive(g)) continue;
                Node& gw = net.nodes[g];
                if (gw.energy <= 0.0) { // assegura consistência
                    deadGateways.insert(g);
                    if (!firstCHFailed) { firstCHFailed = true; firstCHFailureRound = round; }
                    continue;
                }

                int nh = nextHop[g];
                if (nh == -1) {
                    double d = distance(gw.x, gw.y, net.bs.x, net.bs.y);
                    gw.energy -= transmitEnergy(d);
                    if (gw.energy <= 0.0 && deadGateways.find(g) == deadGateways.end()) {
                        deadGateways.insert(g);
                        if (!firstCHFailed) { firstCHFailed = true; firstCHFailureRound = round; }
                    }
                }
                else if (nh >= 0 && nh < net.numGateways && isActive(nh)) {
                    double d = distance(gw.x, gw.y, net.nodes[nh].x, net.nodes[nh].y);
                    gw.energy -= transmitEnergy(d);
                    if (gw.energy <= 0.0 && deadGateways.find(g) == deadGateways.end()) {
                        deadGateways.insert(g);
                        if (!firstCHFailed) { firstCHFailed = true; firstCHFailureRound = round; }
                    }
                    // receiver consumes
                    if (isActive(nh)) {
                        net.nodes[nh].energy -= receiveEnergy();
                        if (net.nodes[nh].energy <= 0.0 && deadGateways.find(nh) == deadGateways.end()) {
                            deadGateways.insert(nh);
                            if (!firstCHFailed) { firstCHFailed = true; firstCHFailureRound = round; }
                        }
                    }
                }
            }

            // --- 3. Atualiza listas de mortos (sensores já marcados, gateways já marcados) ---
            // (já fizemos inserções imediatas quando energy <= 0)

            // --- 4. Sensores órfãos tentam HELP e se reconectam ---
            for (int i = 0; i < net.numSensors; ++i) {
                int sid = net.numGateways + i;
                if (!isSensorAlive(sid)) continue; // sensor morto não busca CH

                int gid = sensorAssignment[i];
                if (isValidGateway(gid)) continue; // já tem CH válido

                int newCH = findNewGateway(sid);
                if (newCH != -1) {
                    sensorAssignment[i] = newCH;
                }
                else {
                    sensorAssignment[i] = -1; // permanece órfão
                }
            }

            // --- 5. Registro por rodada ---
            aliveGws = (int)activeGateways().size();
            aliveSensorsCount = countAliveSensors();
            double avgE_s = avgEnergySensors();
            double avgE_g = avgEnergyGateways();

            csv << round << "," << aliveSensorsCount << "," << aliveGws << ","
                << avgE_s << "," << avgE_g << "," << firstCHFailureRound << "\n";

            ++round;
        } // loop rounds

        csv.close();

        std::cout << "\n=== Fim da simulação ===\n";
        std::cout << "Rodadas executadas: " << round << "\n";
        std::cout << "Gateways ativos restantes: " << activeGateways().size() << "\n";
        if (firstCHFailureRound >= 0) {
            std::cout << "Primeira falha de gateway ocorreu na rodada " << firstCHFailureRound << ".\n";
        }
        else {
            std::cout << "Nenhum CH falhou durante a simulação.\n";
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
