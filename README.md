# FOR-QAOA: Fully Optimized Resource-Efficient QAOA Circuit Simulation

**FOR-QAOA** is a high-performance quantum circuit simulator specifically engineered for simulating the Quantum Approximate Optimization Algorithm (QAOA), targeting large-scale combinatorial optimization problems such as Max-Cut. This simulator achieves remarkable speedups of up to **1490× on GPUs** and **24× on CPUs** compared to state-of-the-art alternatives.

**FOR-QAOA** demonstrates efficient scalability on **multi-node CPU** and **multi-device GPU** architectures, enabling high-fidelity quantum circuit simulations for increasingly complex problems.

## 🔧 Getting Started

### 1. Clone the repository and navigate to the directory.
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Build
If build with single node environment, use make without flags:
```bash
make
```
If build with multi-node environment, use make with USE_MPI=1 flag:
```bash
make USE_MPI=1
```
If build with avx support, use make with USE_AVX=1 flag:
```bash
make USE_AVX=1
```

### 3.  Execute the Simulator
Run the program using the following command:
```bash
./qaoa <P> <N> <D> <C> <B>
```
🧩 Argument Descriptions
 * `<P>`: Circuit depth level (number of QAOA layers).
 * `<N>`: Number of qubits.
 * `<D>`: Log base 2 of the number of devices used for simulation. If you are using $2^D$ devices, set this value to `D`.
 * `<C>`: Chunk size used for cache-local memory within each compute unit.
 * `<B>`: Buffer size for transferring the state vector across devices. ⚠️Constraint: `B ≤ N - D`.

📌 Example: 
```bash
./qaoa 5 30 3 15 27
```
This runs a 5-level QAOA simulation with:

- 30 qubits,
- 8 devices (since 2³ = 8),
- a cache chunk size of 15,
- and a buffer size of 27 for inter-node communication.

### 🌐 Running on a Multi-Node Environment
To run the simulator in a multi-node environment, ensure that you have MPI installed and configured. Use the following command to execute the simulation across multiple nodes:

```bash
mpirun -np <number_of_processes> -bind-to none --map-by ppr:1:node ./qaoa <P> <N> <D> <C> <B>
```
Replace `<number_of_processes>` with the total number of processes you want to run, which should be equal to `2^D` (the number of nodes).

## 🔍 Why Choose FOR-QAOA?
Classical simulation plays a crucial role in validating the performance of QAOA algorithms in ideal, noise-free conditions. However, simulating the full quantum state becomes computationally expensive and memory-intensive as the number of qubits and the QAOA circuit depth increase. FOR-QAOA addresses these challenges by:

- **Achieving High Arithmetic Intensity**: Optimizing computations to maximize the utilization of processing units, leading to compute-bound execution.
- **Minimizing Cache Misses**: Employing strategic data layouts and localized operations to significantly reduce memory access latency.
- **Enabling Efficient Inter-Device Communication**: Leveraging State Vector Transposition (SVT) for optimized data exchange between devices, facilitating better scalability across multi-node and multi-device systems.

## 🧪 How It Works
1. **State Initialization via Launch Control**: Employs an efficient state preparation strategy tailored for QAOA, replacing traditional Hadamard-based initialization.
2. **Compressed Cost-Layer Rotations**: Applies optimized and consolidated rotation gates for the cost Hamiltonian, reducing computational overhead.
3. **Cache-Resident Mixer-Layer RX Gates**: Executes the mixer Hamiltonian's RX gates on localized, cache-resident blocks of the state vector for enhanced performance.
4. **Dynamic State Vector Transposition (SVT)**: Performs state vector transpositions on demand to optimize data locality and facilitate efficient inter-device communication.
5. **Iterative QAOA Layers**: Repeats the cost and mixer layer operations for the specified circuit depth `P`.


## 🧠 Core Techniques
FOR-QAOA achieves its high performance through the following innovative techniques that push the simulation towards being compute-bound:

1. **Cache-Resident State Operations**: Improves data locality by dividing the global state vector into smaller, manageable chunks that fit within the cache, enabling efficient in-cache computations.

2. **State Vector Transposition (SVT)**: Facilitates efficient cross-device communication and ensures optimal data alignment across devices, minimizing communication overhead and maximizing data locality.

3. **Launch Control**: Replaces the standard Hadamard-based initial state preparation with a more efficient method specifically designed for the structure of QAOA circuits.

4. **Rotation Compression**: Consolidates consecutive rotation gates within the cost layer into a smaller number of optimized operations, thereby reducing the overall simulation overhead.

## 📈 Performance
Tested on:
- **8× DGX-H100 servers (64 GPUs total) for GPU scalability**
- **8× multi-node CPU clusters (Intel Xeon Platinum 8352V / 8480+)**

✅ Outperforms Qiskit-Aer, cuQuantum, mpiQulacs, QuEST, and UniQ\
✅ Scales efficiently up to 64 GPUs and 8 CPU nodes\
✅ Maintains performance across both strong and weak scaling scenarios

## 📚 Reference
**FOR-QAOA: Fully Optimized Resource-Efficient QAOA Circuit Simulation for Solving the Max-Cut Problems**\
PEARC’25 (Practice and Experience in Advanced Research Computing)
[Paper DOI – https://doi.org/10.1145/3708035.3736006]