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
 
#ifndef _RZZGATE_H_
#define _RZZGATE_H_

#include <cmath>
#include "state.h"

void rotationCompressionUnweighted(Qureg qureg, double angle, ull *graph, bool isFirstLayer);
void rotationCompressionWeighted(Qureg qureg, double angle, bool *graph, double *weights, bool isFirstLayer);

#endif