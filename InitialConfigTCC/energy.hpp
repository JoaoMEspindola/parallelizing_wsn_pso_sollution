#ifndef ENERGY_HPP
#define ENERGY_HPP

#include <cmath>

// Parâmetros de energia
constexpr double E_ELEC = 50e-9;       // Energia para operar a eletrônica (J/bit)
constexpr double EPS_FS = 10e-12;      // Amplificador free space (J/bit/m^2)
constexpr double EPS_MP = 0.0013e-12;  // Amplificador multipath (J/bit/m^4)
constexpr double D_0 = 87.0;           // Distância limite entre modelos
constexpr int PACKET_SIZE = 4000;      // bits por pacote

// Energia para transmitir dados a uma distância d
inline double transmitEnergy(double d) {
    if (d < D_0)
        return PACKET_SIZE * (E_ELEC + EPS_FS * d * d);
    else
        return PACKET_SIZE * (E_ELEC + EPS_MP * std::pow(d, 4));
}

// Energia para receber um pacote
inline double receiveEnergy() {
    return PACKET_SIZE * E_ELEC;
}

#endif
