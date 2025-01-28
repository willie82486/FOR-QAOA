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