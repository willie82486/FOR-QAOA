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

#include "state.h"

ull* init1DGraph(const State& qureg);
void free1DGraph(ull* graph);
void addEdgeTo1DGraph(const State& qureg, ull* graph, int qubit1, int qubit2);

Fp* init2DGraph(const State& qureg);
void addEdgeTo2DGraph(const State& qureg, Fp* graph, int qubit1, int qubit2, Fp weight);
void free2DGraph(Fp* graph);

#endif