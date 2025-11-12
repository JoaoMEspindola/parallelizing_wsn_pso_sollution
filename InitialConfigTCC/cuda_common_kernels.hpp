#pragma once
#include <curand_kernel.h>

__global__ void initCurandKernel(curandState* states, unsigned long seed, int n);
