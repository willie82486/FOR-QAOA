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
 
#ifndef _RXGATE_H_
#define _RXGATE_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include "state.h"
#include "sqsGate.h"
#include "csqsGate.h"


ull bit_string(const ull task, const int target);
void rotateX(Qureg& qureg, const double angle, const int targetQubit, ull idx);
void multiRotateX(Qureg& qureg, const double angle, 
                std::vector<std::vector<int>>& subCirs, bool csqsflag, std::vector<int>& qubitMap);
void multiRotateX_avx(Qureg& qureg, const double angle, 
                std::vector<std::vector<int>>& subCirs, bool csqsflag, std::vector<int>& qubitMap);
    
#endif