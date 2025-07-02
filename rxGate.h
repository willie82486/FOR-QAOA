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