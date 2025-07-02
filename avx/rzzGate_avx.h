#ifndef RZZ_GATE_AVX_H
#define RZZ_GATE_AVX_H

#include <cassert>
#include <cmath>

#include "../state.h"

void rotationCompressionUnweighted_avx(Qureg qureg, double angle, ull *graph, bool isFirstLayer);
void rotationCompressionWeighted_avx(Qureg qureg, double angle, bool *graph, double *weights, bool isFirstLayer);

#endif