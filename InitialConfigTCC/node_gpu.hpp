// node_gpu.hpp
#pragma once

// Estrutura simples POD para copiar para o device.
// Usamos atributos __host__ __device__ para permitir construções/leituras
// em ambos ambientes (útil para testes e inicialização inline).
struct NodeGPU {
    double x;
    double y;
    double energy;
    int id;         // opcional: id do nó (0..N-1)
    int isGateway;  // 1 se gateway, 0 se sensor (evita bool em device structs)

    __host__ __device__
        NodeGPU() : x(0.0), y(0.0), energy(0.0), id(-1), isGateway(0) {}

    __host__ __device__
        NodeGPU(double _x, double _y, double _energy, int _id = -1, int _isG = 0)
        : x(_x), y(_y), energy(_energy), id(_id), isGateway(_isG) {
    }
};
