### *CUDA-based Parallel PSO for Wireless Sensor Networks*

This repository contains the full implementation of the routing + unequal clustering solution for Wireless Sensor Networks (WSN) proposed by **Azharuddin & Jana (2016)**.
It includes:

* A **sequential CPU implementation** (C++17)
* A **parallel GPU implementation using one thread per particle**
* A **parallel GPU implementation using one block per swarm**
* Utilities, common CUDA kernels, simulation modules, and export tools

The main goal is to evaluate how different CUDA parallelization granularities affect the performance and scalability of Particle Swarm Optimization (PSO) when applied to large-scale WSN lifetime maximization.

---

## **Project Structure**

```
InitialConfigTCC/
│
├── Header Files (.hpp)
│   ├── cluster_radius.hpp
│   ├── clustering_pso.hpp
│   ├── clustering_pso_cuda.hpp
│   ├── clustering_pso_cuda_blocks.hpp
│   ├── compact_graph.hpp
│   ├── cuda_common_kernels.hpp
│   ├── energy.hpp
│   ├── energy_gpu.hpp
│   ├── export.hpp
│   ├── network.hpp
│   ├── node.hpp
│   ├── node_gpu.hpp
│   ├── routing_pso.hpp
│   ├── routing_pso_block_cuda.hpp
│   ├── routing_pso_cuda.hpp
│   ├── simulation.hpp
│   └── utils.hpp
│
├── Source Files (.cpp / .cu)
│   ├── clustering_pso_cuda.cu
│   ├── clustering_pso_cuda_blocks.cu
│   ├── cuda_common_kernels.cu
│   ├── export.cpp
│   ├── main.cpp
│   ├── routing_pso_cuda.cu
│   ├── routing_pso_block_cuda.cu
│   ├── utils.cpp
│
└── plotting/  (optional Python scripts for figures)
```

---

## **Requirements**

### **Hardware**

* NVIDIA GPU with **Compute Capability ≥ 6.1**
  (validated on **GTX 1050 Ti**, Pascal architecture)
* x86_64 CPU
* 16 GB RAM recommended

### **Software**

* **NVIDIA CUDA Toolkit 12.x**
* **Visual Studio 2022 + MSVC 19.36**
* **Python 3.10+** (for data visualization)
* (Optional) **CMake** for external builds

---

## **How to Build (Visual Studio 2022)**

1. Install **CUDA Toolkit 12.x**
2. Open Visual Studio → *Open Folder* → select `InitialConfigTCC/`
3. Ensure `.cu` files are recognized as **CUDA C/C++**
4. Build using:
   **Build → Build Solution**
5. Run using:
   **Debug → Start Without Debugging**

Visual Studio automatically calls `nvcc` behind the scenes using project configuration settings.

---

Output files (CSV) are generated into:

```
outputs/
    gbest_timelines/
    execution_times/
    fitness_logs/
```

---

## **Plotting Scripts (Python)**

Example scripts:

```
python plotting/plot_gbest_timeline.py
python plotting/plot_gpu_vs_gpu_target.py
```

These produce the figures used in the experimental analysis.

---

## **Execution Pipeline Overview**

```
main.cpp
   → routing_pso_*   (PSO-based routing step)
   → clustering_pso_* (unequal clustering computation)
   → simulation.hpp   (lifetime evaluation and metrics)
   → export.cpp       (CSV output)
```

---

## **Results Summary**

* GPU implementations achieved **up to 32× speedup** over CPU.
* GPU Blocks strategy consistently outperformed GPU Threads.
* Parallel PSO allowed the WSN optimization problem to scale to large networks.
* The parallel approach maintained solution quality while drastically reducing execution time.

---

## **Reference**

```bibtex
@misc{Espindola2025_parallelizing_wsn_pso,
  author = {Espíndola, Joao M.},
  title = {parallelizing_wsn_pso_sollution: CUDA-parallelized PSO for WSN lifetime optimization},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/JoaoMEspindola/parallelizing_wsn_pso_sollution}},
  note = {accessed 2025-12-08}
}
```
