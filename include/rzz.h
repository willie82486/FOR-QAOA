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
 
#ifndef _RZZ_H_
#define _RZZ_H_

#include "state.h"

// extern __managed__ Fp initialAmp;

void initH(const State& qureg);
void rotationCompressionUnweighted(const State& qureg, Fp angle, ull* graph, bool isFirstLayer);
void rotationCompressionWeighted(State& qureg,const Fp angle,const __restrict__ Fp* graph,const bool isFirstLayer);

#endif