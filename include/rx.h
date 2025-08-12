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
 
#ifndef _RX_H_
#define _RX_H_
#include <iostream>
#include <vector>
#include <cassert>
#include "state.h"
using namespace std;

void rotateX(const State& qureg,const int targetQubit,const Fp angle);
void multiRotateX(State& qureg, vector<int> &targetQubits,const int gate_size,const Fp angle);

#endif