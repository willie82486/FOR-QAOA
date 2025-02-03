#include "graph.h"

#include <assert.h>

#include "helper_cuda.h"

ull* init1DGraph(const State& qureg) {
    ull* graph;
    size_t size = qureg.numQubits * sizeof(ull);
    checkCudaErrors(cudaMallocManaged(&graph, size));
    memset(graph, 0, size);
    return graph;
}

void free1DGraph(ull* graph) {
    checkCudaErrors(cudaFree(graph));
}

void addEdgeTo1DGraph(const State& qureg, ull* graph, int qubit1, int qubit2) {
    assert(qubit1 < qureg.numQubits);
    assert(qubit2 < qureg.numQubits);
    if (qubit1 > qubit2) {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1] |= 1ull << qubit2;
}

Fp* init2DGraph(const State& qureg) {
    Fp* graph;
    size_t size = qureg.numQubits * qureg.numQubits * sizeof(Fp);
    checkCudaErrors(cudaMallocManaged(&graph, size));
    memset(graph, 0, size);
    return graph;
}

void addEdgeTo2DGraph(const State& qureg, Fp* graph, int qubit1, int qubit2, Fp weight) {
    int numQubits = qureg.numQubits;
    assert(qubit1 < numQubits);
    assert(qubit2 < numQubits);
    if (qubit1 > qubit2) {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1*numQubits+qubit2] = weight;
}

void free2DGraph(Fp* graph) {
    checkCudaErrors(cudaFree(graph));
}