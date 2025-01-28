#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "state.h"

ull* init1DGraph(const State& qureg);
void free1DGraph(ull* graph);
void addEdgeTo1DGraph(const State& qureg, ull* graph, int qubit1, int qubit2);

Fp* init2DGraph(const State& qureg);
void addEdgeTo2DGraph(const State& qureg, Fp* graph, int qubit1, int qubit2, Fp weight);
void free2DGraph(Fp* graph);

#endif