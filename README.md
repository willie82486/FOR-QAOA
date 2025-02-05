# FOR-QAOA_GPU
Simulate the Quantum Approximate Optimization Algorithm (QAOA) on a classical computer using a GPU platform.

## Getting started

### 1. Clone this repository and navigate to the corresponding directory.

### 2. Build
```bash
make
```
### 3. Execute ./weighted to run the program. The output will display the elapsed time based on the parameters provided.
```bash
./weighted P N D C B
```
 where 
 * `<P>` refers to the circuit depth level.
 * `<N>` refers to the number of qubits
 * `<D>` refers to the number of devices used to simulate QAOA. If you have N devices, then N = 2^D.
 * `<C>` refers to the chunk size allocated as cache-like memory.
 * `<B>` refers to the buffer size used for transferring the state vector between local and other devices. The constraint is that B must be ≤ N - D.
- Example: 
```bash
./weighted 5 30 3 12 27
```
