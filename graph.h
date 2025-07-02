#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <cstdlib>
#include <cassert>
#include "state.h"

ull* init1DGraph(Qureg qureg);
void addEdgeTo1DGraph(Qureg qureg, ull* graph, int qubit1, int qubit2);
void free1DGraph(ull* graph);
bool* init2DGraph(Qureg qureg);
double* initWeights(Qureg qureg);
void freeWeights(double* weights);
void addEdgeTo2DGraph(Qureg qureg, bool* graph, double* weights, 
                    int qubit1, int qubit2, double weight);
void free2DGraph(bool *graph);
#endif