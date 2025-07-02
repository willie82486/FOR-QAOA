#ifndef RX_GATE_AVX_H
#define RX_GATE_AVX_H

#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include "../state.h"
#include "../sqsGate.h"
#include "../csqsGate.h"

ull bit_string(const ull task, const int target);
void rotateX_avx(Qureg& qureg, const double angle, const int targetQubit, ull idx);
void multiRotateX_avx(Qureg& qureg, const double angle, 
    std::vector<std::vector<int>>& subCirs, bool csqsflag, std::vector<int>& qubitMap);

#endif