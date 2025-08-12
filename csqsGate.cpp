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
 
#include "csqsGate.h"

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
void CSQS(Qureg& qureg, const int csqsSize, char* targ) {
    MPI_Status status;
    const int &N = qureg.numQubitTotal;
    const int &D = qureg.numQubitTotal - qureg.numQubitPerRank;
    const int &B = qureg.numQubitPerBuffer;

    const int numGroups = 1 << (D - csqsSize);
    const int numMemberInGroup = 1 << csqsSize;
    std::vector<std::vector<int>> devlist(numGroups, std::vector<int>(numMemberInGroup));
    for (int mbr = 0; mbr < numMemberInGroup; mbr++) {
        int mem_bits = 0;
        for (int i = 0; i < csqsSize; i++) 
            mem_bits |= ((mbr >> i) & 1) << targ[i];
        devlist[0][mbr] = mem_bits;
    }

    for (int grp = 1; grp < numGroups; grp++) {
        int grp_bits = grp;
        for (int i = 0; i < csqsSize; i++)
            grp_bits = insert_bits_0<1>(grp_bits, &targ[i]);
        for (int mbr = 0; mbr < numMemberInGroup; mbr++)
            devlist[grp][mbr] = grp_bits | devlist[0][mbr];
    }
    
    for (ull off  = 0; off < (1ull << (N - D - csqsSize)); off += (1ull << (B - csqsSize))) {
        for (ull grp = 0; grp < (1ull << (D - csqsSize)); grp++) {
            // 用mask表達以避免同一時間多台機器(rank: 1, 2, 3) 對單一機器(rank: 0) 做傳輸
            for (ull mask = 1; mask < (1ull << csqsSize); mask++) {
                for (ull mbrAInGroup = 0; mbrAInGroup < (1ull << csqsSize); mbrAInGroup++) {
                    if (devlist[grp][mbrAInGroup] == qureg.rankId) {
                        int mbrBInGroup = mbrAInGroup ^ mask;
                        int devB = devlist[grp][mbrBInGroup];
                        if (qureg.rankId == devB) continue;
                        ull off_sv = mbrBInGroup * (1ull << (N - D - csqsSize)) + off;
                        ull off_bf = mbrBInGroup * (1ull << (B - csqsSize));
                        // printf("rankId: %d, off_sv: %d, off_bf: %d\n", qureg.rankId, off_sv, off_bf);
                        MPI_CHECK(MPI_Sendrecv(qureg.stateVec + off_sv, 1 << (B - csqsSize), MPI_C_DOUBLE_COMPLEX, devB, 0,
                            qureg.buffer + off_bf, 1 << (B - csqsSize), MPI_C_DOUBLE_COMPLEX, devB, 0, MPI_COMM_WORLD, &status));
                        memcpy(qureg.stateVec + off_sv, qureg.buffer + off_bf, (1 << (B - csqsSize)) * sizeof(Complex));
                    }
                }
            }
        }
    }
}



void CSQS_MultiThread(Qureg &qureg, const int csqsSize, char* targ)
{
    // MPI_Status status;
    const int &N = qureg.numQubitTotal;
    const int &D = qureg.numQubitTotal - qureg.numQubitPerRank;
    const int &B = qureg.numQubitPerBuffer;
    const int &T = qureg.numQubitThreads;

    const int numGroups        = 1 << (D - csqsSize);
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
                        int off_thread = (volTransmission_thread * thread_id);

                        for (ull off = 0; off < volTransmission_total; off += volTransmission_unit) {
                            ull off_sv = mbrBInGroup * volTransmission_total + off;
                            ull off_bf = mbrBInGroup * volTransmission_unit;
                            
                            // MPI_Status status;
                            // MPI_Sendrecv(state.selfStateVec + off_sv + off_thread, volTransmission_thread, MPI_C_DOUBLE_COMPLEX, devB, thread_id,
                            //                 state.selfBuffer + off_bf + off_thread, volTransmission_thread, MPI_C_DOUBLE_COMPLEX, devB, thread_id, MPI_COMM_WORLD, &status);

                            MPI_Request sendRequest;
                            MPI_Request recvRequest;
                            MPI_Isend(qureg.stateVec + off_sv + off_thread, volTransmission_thread, MPI_C_DOUBLE_COMPLEX, devB, thread_id, MPI_COMM_WORLD, &sendRequest);
                            MPI_Irecv(qureg.buffer + off_bf + off_thread, volTransmission_thread, MPI_C_DOUBLE_COMPLEX, devB, thread_id, MPI_COMM_WORLD, &recvRequest);

                            // while (true) {
                            //     int sendFlag = 0, recvFlag = 0;
                            //     MPI_Test(&sendRequest, &sendFlag, MPI_STATUS_IGNORE);
                            //     MPI_Test(&recvRequest, &recvFlag, MPI_STATUS_IGNORE);
                            //     if (sendFlag && recvFlag) break;
                            // }
                           
                            
                            MPI_Wait(&sendRequest, MPI_STATUS_IGNORE);
                            MPI_Wait(&recvRequest, MPI_STATUS_IGNORE);

                            // MPI_Cancel(&sendRequest);
                            // MPI_Cancel(&recvRequest);
                            // MPI_Request_free(&sendRequest);
                            // MPI_Request_free(&recvRequest);

                            memcpy(qureg.stateVec + off_sv + off_thread, qureg.buffer + off_bf + off_thread, volTransmission_thread * sizeof(Complex));
                            // copy(state.selfBuffer + off_bf + off_thread, state.selfBuffer + off_bf + off_thread + volTransmission_thread, state.selfStateVec + off_sv + off_thread);
                            // memmove(state.selfStateVec + off_sv + off_thread, state.selfBuffer + off_bf + off_thread, volTransmission_thread * sizeof(Complex));
                        }
                    }        
                }
            }
        }
    }
}
#endif