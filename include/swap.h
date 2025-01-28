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