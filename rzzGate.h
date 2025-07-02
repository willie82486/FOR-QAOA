#ifndef _RZZGATE_H_
#define _RZZGATE_H_

#include <cmath>
#include "state.h"

void rotationCompressionUnweighted(Qureg qureg, double angle, ull *graph, bool isFirstLayer);
void rotationCompressionWeighted(Qureg qureg, double angle, bool *graph, double *weights, bool isFirstLayer);

#endif