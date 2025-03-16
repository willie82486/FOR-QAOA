# FOR-QAOA_GPU
**FOR-QAOA** is a high-performance quantum circuit simulator specifically designed for simulating the Quantum Approximate Optimization Algorithm (QAOA), targeting large-scale combinatorial optimization problems such as Max-Cut. This simulator achieves up to **1490× GPU speedup** and **24× CPU speedup** compared to existing state-of-the-art quantum simulators.

**FOR-QAOA** efficiently runs on **multi-node CPUs** and **multi-device GPUs**, enabling scalable, high-fidelity quantum circuit simulations.

## 🔧 Getting started

### 1. Clone this repository and navigate to the corresponding directory.

### 2. Build
```bash
make
```
In the `makefile`,
* the `-arch` flag in `NVCXXFLAGS` may need to be modified based on the GPU you are using.
  
### 3. Execute `./weighted` to Run the Simulator
Run the program with the following command:
```bash
./weighted P N D C B
```
🧩 Argument Descriptions
 * `<P>`: Circuit depth level (number of QAOA layers).
 * `<N>`: Number of qubits.
 * `<D>`: Number of devices used for simulation. If you have `N` devices, then `N = 2^D`.
 * `<C>`: Chunk size used as cache-local memory in each compute unit.
 * `<B>`: Buffer size for transferring the state vector across devices. ⚠️The constraint is that B must be ≤ N - D.

📌 Example: 
```bash
./weighted 5 30 3 12 27
```
This runs a 5-level QAOA simulation with:

- 30 qubits,
- 8 devices (since 2³ = 8),
- a cache chunk size of 12,
- and a buffer size of 27 for inter-device communication.

### 🌐 Running on a Multi-Node Environment
If you're using a distributed setup (e.g., 2 nodes with 8 GPUs each), use mpirun like so:
```bash
mpirun -np 16 -bind-to none --map-by ppr:8:node ./weighted 5 30 4 12 26
```
- `-np 16`: Total number of processes (e.g., 16 GPUs = 2 nodes × 8 GPUs).
- `--map-by ppr:8:node`: Map 8 processes per node.
- `4` in the command means `2⁴ = 16` devices.

## 🔍 Why FOR-QAOA?
Classical simulation of QAOA circuits is vital for validating algorithm performance in a noise-free environment. However, full-state simulations are memory- and compute-intensive, especially as the number of qubits and QAOA depth grows. FOR-QAOA tackles this challenge with:

- **High arithmetic intensity**, pushing simulation into compute-bound territory.
- **Reduced cache misses**, thanks to strategic data layout and localized operations.
- **Efficient inter-device communication**, enabled by SVT, allowing better scaling across nodes/devices.

## 🧪 How It Works
1. Initializes the quantum state using Launch Control.
2. Applies compressed cost-layer rotations.
3. Executes mixer-layer RX gates using cache-resident blocks.
4. Dynamically performs state vector transposition if needed 5. to optimize data locality.
5. Repeats the QAOA layers for the configured depth `P`.


## 🧠 Core Techniques
The following method push the algorithm to be compute bound.
1. `cache-resident state operations`:  Improves cache locality by partitioning the global state vector into small, localized chunks for efficient in-cache execution.
2. `state vector transposition (SVT)`: Enables cross-device communication and data alignment for optimal state vector layout and minimal communication overhead.
3. `launch control`: Replaces Hadamard-based state initialization with a more efficient preparation strategy tailored for QAOA circuits.
4. `rotation compression` Consolidates multiple rotation gates in the cost layer into fewer, optimized operations to reduce simulation overhead.

## 📈 Performance
Tested on:
- **8× DGX-H100 servers (64 GPUs total) for GPU scalability**
- **8× multi-node CPU clusters (Intel Xeon Platinum 8352V / 8480+)**\

✅ Outperforms Qiskit-Aer, cuQuantum, mpiQulacs, QuEST, and UniQ\
✅ Scales efficiently up to 64 GPUs and 8 CPU nodes\
✅ Maintains performance across both strong and weak scaling scenarios\

## 📚 Reference
**FOR-QAOA: Fully Optimized Resource-Efficient QAOA Circuit Simulation for Solving the Max-Cut Problems**\
PEARC’25 (Practice and Experience in Advanced Research Computing)
[Paper DOI – TBD]
