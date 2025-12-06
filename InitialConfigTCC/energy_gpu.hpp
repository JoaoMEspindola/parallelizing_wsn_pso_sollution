#ifndef ENERGY_GPU_HPP
#define ENERGY_GPU_HPP

// ===============================
// Funções auxiliares GPU inline
// ===============================
#ifdef __CUDACC__   // <-- só compilar se NVCC estiver compilando

// distância GPU
__device__ __forceinline__
double distanceGPU(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

// energia de transmissão
__device__ __forceinline__
double transmitEnergyGPU(double d) {
    const double E_elec = 50e-9;
    const double E_amp = 10e-12;
    const double k = 4000.0;
    return (E_elec + E_amp * d * d) * k;
}

// energia de recepção
__device__ __forceinline__
double receiveEnergyGPU() {
    const double E_elec = 50e-9;
    const double k = 4000.0;
    return E_elec * k;
}
#endif // __CUDACC__

#endif
