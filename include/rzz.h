#ifndef _RZZ_H_
#define _RZZ_H_

#include "state.h"

// extern __managed__ Fp initialAmp;

void initH(const State& qureg);
void rotationCompressionUnweighted(const State& qureg, Fp angle, ull* graph, bool isFirstLayer);
void rotationCompressionWeighted(State& qureg,const Fp angle,const __restrict__ Fp* graph,const bool isFirstLayer);

#endif