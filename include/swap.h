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
 
#ifndef _SWAP_H_
#define _SWAP_H_
#include <iostream>
#include <vector>
#include <cassert>
#include "state.h"
using namespace std;
void mapping_after_swap_conti(const State& qureg, Fp*  graph,const int SwapOut,const int SwapIn,const int numSwap);
void mapping_after_swap_conti(const State& qureg, ull* graph,const int SwapOut,const int SwapIn,const int numSwap);

void swap_gate(const State& qureg,const int targ0,const int targ1);
void swap_gate_conti(const State& qureg,const int SwapOut,const int SwapIn,const int numSwap);

#endif