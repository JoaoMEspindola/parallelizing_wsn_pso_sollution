#ifndef NODE_HPP
#define NODE_HPP

#include <vector>

struct Node {
    int id;
    double x, y;
    bool isGateway;
    double energy;
    double initialEnergy;
    std::vector<int> neighbors;

    Node(int id_, double x_, double y_, bool isGateway_, double energy_)
        : id(id_), x(x_), y(y_), isGateway(isGateway_), energy(energy_), initialEnergy(energy_) {
    }
};

#endif
