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
 
#include "state.h"

Qureg createStateVec(int N, int D, int B, int T, int C, int rankId) {
    Qureg qureg;
    qureg.numQubitTotal = N;
    qureg.numQubitPerRank = (N - D);
    qureg.numQubitPerBuffer = B;
    qureg.numQubitPerChunk = C;

    qureg.numAmpTotal = 1ull << N;
    qureg.numAmpPerRank = 1ull << (N - D);
    qureg.numAmpPerBuffer = 1ull << B;
    qureg.numAmpPerChunk = 1ull << C;
    
    qureg.numQubitThreads = T;
    qureg.numThreads = 1ull << T;

    qureg.stateSizePerRank = qureg.numAmpPerRank * sizeof(Complex);
    qureg.numRanks = 1 << D;
    qureg.rankId = rankId;
    qureg.stateVec = (Complex *)calloc(qureg.numAmpPerRank, sizeof(Complex));
    if (qureg.numRanks > 1) {
        qureg.buffer = (Complex *)calloc(qureg.numAmpPerBuffer, sizeof(Complex));
    }
#ifdef USE_MPI
    int csqsSize = D;
    char *targ = (char *)calloc(csqsSize, sizeof(char));
    for(int i = 0; i < csqsSize; i++) {
        targ[i] = i;
    }
    MPI_Warmup(qureg, csqsSize, targ);
#endif
    return qureg;
}

uint64_t __attribute__((always_inline)) inline insert_bit_0(uint64_t task, const char targ) {
    uint64_t mask = (1ULL << targ) - 1;
    return ((task >> targ) << (targ + 1)) | (task & mask);
}

template<int N> 
uint64_t __attribute__((always_inline)) inline insert_bits_0(uint64_t task, const char targs[]) {
    for (int i = 0; i < N; i++)
        task = insert_bit_0(task, targs[i]);
    return task;
}
#ifdef USE_MPI
void MPI_Warmup(Qureg& qureg, const int csqsSize, char* targ) {
    const int &N = qureg.numQubitTotal;
    const int &D = qureg.numQubitTotal - qureg.numQubitPerRank;
    const int &B = qureg.numQubitPerBuffer;
    const int &T = qureg.numQubitThreads;

    const int numGroups = 1 << (D - csqsSize);
    const int numMemberInGroup = 1 << csqsSize;
    std::vector<std::vector<ull>> devlist(numGroups, std::vector<ull>(numMemberInGroup));
    for (int mbr = 0; mbr < numMemberInGroup; mbr++) {
        ull mem_bits = 0;
        for (int i = 0; i < csqsSize; i++) 
            mem_bits |= ((mbr >> i) & 1) << targ[i];
        devlist[0][mbr] = mem_bits;
    }

    for (int grp = 1; grp < numGroups; grp++) {
        ull grp_bits = grp;
        for (int i = 0; i < csqsSize; i++)
            grp_bits = insert_bits_0<1>(grp_bits, &targ[i]);
        for (int mbr = 0; mbr < numMemberInGroup; mbr++)
            devlist[grp][mbr] = grp_bits | devlist[0][mbr];
    }

    ull volTransmission_total  = (1 << (N - D - csqsSize));
    ull volTransmission_unit   = (1 << (B - csqsSize));
    ull volTransmission_thread = (1 << (B - csqsSize)) >> T;
    int numMbr_in_grp          = (1 << csqsSize);
    int numGrp                 = (1 << (D - csqsSize));

    for (int grp = 0; grp < numGrp; grp++) {
        // 用mask表達以避免同一時間多台機器(rank: 1, 2, 3) 對單一機器(rank: 0) 做傳輸
        // 之後在4台以上的機器與原本的版本比較
        for (int mask = 1; mask < numMbr_in_grp; mask++) {       
            for (int mbrAInGroup = 0; mbrAInGroup < numMbr_in_grp; mbrAInGroup++) {
                if (devlist[grp][mbrAInGroup] == qureg.rankId) {
                    int mbrBInGroup = mbrAInGroup ^ mask;
                    int devB = devlist[grp][mbrBInGroup];
                    if (qureg.rankId == devB) continue;
                    
                    # pragma omp parallel for schedule(dynamic) num_threads(qureg.numThreads)
                    for (int thread_id = 0; thread_id < qureg.numThreads; thread_id++) {
                        ull off_thread = (volTransmission_thread * thread_id);

                        for (ull off = 0; off < volTransmission_total; off += volTransmission_unit) {
                            ull off_sv = mbrBInGroup * volTransmission_total + off;
                            ull off_bf = mbrBInGroup * volTransmission_unit;
                            
                            MPI_Status status;
                            MPI_Sendrecv(qureg.stateVec + off_sv + off_thread, volTransmission_thread, MPI_C_DOUBLE_COMPLEX, devB, thread_id,
                                           qureg.buffer + off_bf + off_thread, volTransmission_thread, MPI_C_DOUBLE_COMPLEX, devB, thread_id, MPI_COMM_WORLD, &status);
                        }
                    } 
                }
            }
        }
    }
}
#endif

void printStateVec(Qureg& qureg, const int printNum) {
    for (int i = 0; i < printNum; i++) {
        printf("rank: %d StateVector[%d]: %f + %fi\n", qureg.rankId, i, qureg.stateVec[i].real, qureg.stateVec[i].imag);
    }
    return;
}

void destroyStateVec(Qureg& qureg)
{
    free(qureg.stateVec);
    if (qureg.numRanks > 1) {
        free(qureg.buffer);
    }
    return;
}