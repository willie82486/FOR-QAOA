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
 
#include "rxGate.h"

inline ull bit_string(const ull task, const int target) {
    ull mask = (1ull << target) - 1;
    return ((task >> target) << (target + 1)) | (task & mask);
}

inline void sincosf64(double angle, double* sinRad, double* cosRad) {
    *sinRad = sin(angle);
    *cosRad = cos(angle);
}

void rotateX(Qureg& qureg, const double angle, const int targetQubit, ull idx) {
    double sinAngle, cosAngle;
    sincosf64(angle / 2, &sinAngle, &cosAngle);
    ull half_off = 1 << targetQubit, off = half_off * 2;
    ull off0 = idx, off1 = idx + half_off;
    Complex up, lo;
    for (ull i = 0; i < qureg.numAmpPerChunk; i += off) {
        for (ull j = 0; j < half_off; j++) {
            up = qureg.stateVec[off0];
            lo = qureg.stateVec[off1];
            qureg.stateVec[off0].real =  up.real * cosAngle + lo.imag * sinAngle;
            qureg.stateVec[off0].imag =  up.imag * cosAngle - lo.real * sinAngle;
            qureg.stateVec[off1].real =  up.imag * sinAngle + lo.real * cosAngle;
            qureg.stateVec[off1].imag = -up.real * sinAngle + lo.imag * cosAngle;
            off0 += 1;
            off1 += 1;
        }
        off0 += half_off;
        off1 += half_off;
    }
}

void multiRotateX(Qureg& qureg, const double angle, 
                    std::vector<std::vector<int>>& GBs, bool csqsflag, std::vector<int>& qubitMap) {
    for (size_t GBsIdx = 0; GBsIdx < GBs.size(); GBsIdx++) {
#ifdef USE_MPI
        if (GBsIdx > 0 && (*GBs[GBsIdx-1].rbegin() == qureg.numQubitPerRank-1) && csqsflag) {
            int csqsSize = qureg.numQubitTotal - qureg.numQubitPerRank; // Device qubit
            char *targ = (char *)calloc(csqsSize, sizeof(char));
            for (int i = 0; i < csqsSize; i++) {
                targ[i] = i;
            }
            CSQS_MultiThread(qureg, csqsSize, targ);
            csqsflag = false;
            for (int i = qureg.numQubitTotal-csqsSize; i < qureg.numQubitTotal; i++) {
                std::swap(qubitMap[i-csqsSize], qubitMap[i]);
            }
        }
#endif
        if (GBsIdx != 0) {
            sqs_info_t info;
            initSQSInfo(qureg, info, GBs[GBsIdx]);
            execSQS(qureg.stateVec, info);
            int swapSize = GBs[GBsIdx].size();
            for (int i = 0; i < swapSize; i++) {
                std::swap(qubitMap[qureg.numQubitPerChunk-1-i], qubitMap[GBs[GBsIdx][swapSize-1-i]]);
            }
        }   
     
        #pragma omp parallel for schedule(static)
        for (ull idx = 0; idx < qureg.numAmpPerRank; idx += qureg.numAmpPerChunk) {
            for (int i = qureg.numQubitPerChunk-GBs[GBsIdx].size() ; i < qureg.numQubitPerChunk; i++) {
                rotateX(qureg, angle, GBs[0][i], idx);
            }
        }   
    }
}
