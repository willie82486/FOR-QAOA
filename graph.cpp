/*
 * Copyright 2025 The FOR-QAOA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#include "graph.h"


ull* init1DGraph(Qureg qureg) {
    return (ull *)calloc(qureg.numQubitTotal, sizeof(ull));
}

void addEdgeTo1DGraph(Qureg qureg, ull* graph, int qubit1, int qubit2) {
    int numQubits = qureg.numQubitTotal;
    assert(qubit1 < numQubits);
    assert(qubit2 < numQubits);
    if (qubit1 > qubit2) {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1] |= (1ull << qubit2);
}

void free1DGraph(ull* graph) {
    free(graph);
}

bool *init2DGraph(Qureg qureg)
{
    int numQubits = qureg.numQubitTotal;
    return (bool *)calloc(numQubits * numQubits, sizeof(bool));
}

double *initWeights(Qureg qureg)
{
    int numQubits = qureg.numQubitTotal;
    return (double *)malloc(numQubits * numQubits * sizeof(double));
}

void freeWeights(double *weights)
{
    free(weights);
}

void addEdgeTo2DGraph(Qureg qureg, bool *graph, double *weights, int qubit1, int qubit2, double weight)
{
    int numQubits = qureg.numQubitTotal;
    assert(qubit1 < numQubits);
    assert(qubit2 < numQubits);
    if (qubit1 > qubit2)
    {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1 * numQubits + qubit2] = true;
    weights[qubit1 * numQubits + qubit2] = weight;
}

void free2DGraph(bool *graph)
{
    free(graph);
}
