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