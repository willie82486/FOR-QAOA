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
 
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "state.h"
#include "graph.h"
#include "rzzGate.h"
#include "avx/rzzGate_avx.h"
#include "avx/rxGate_avx.h"
#include "rxGate.h"
#include "utils.h"
#include <chrono>

double H;

void initH(Qureg qureg) {
    int numQubits = qureg.numQubitTotal;
    H = powf(1.f / sqrt(2.f), (double)numQubits);
}

int main(int argc, char *argv[]) {
    int rank = 0, size = 1, provided;
#ifdef USE_MPI
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
        exit(-1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

#ifdef USE_AVX
    printf("AVX512 is enabled\n");
#endif
    int P, N, D, B, C, T;
    if (argc >= 7) {
        P = strtol(argv[1], NULL, 10);
        N = strtol(argv[2], NULL, 10);
        D = strtol(argv[3], NULL, 10);
        C = strtol(argv[4], NULL, 10);
        T = strtol(argv[5], NULL, 10);
        B = strtol(argv[6], NULL, 10);
    }
    else {
        P = 1;
        N = 30;
        D = 3;
        C = 16;
        T = 4;
        B = 22;
    }
    if (rank == 0) {
        printf("P: %d N: %d D: %d C: %d T: %d B: %d\n", P, N, D, C, T, B);
    }

    const double beta = M_PI_4, gamma = M_PI_4;

    std::vector<std::vector<int>> GBs;
    for (int i = 0; i < N - D; i += C) {
        std::vector<int> GB;
        for (int j = i; j < i + C && j < N - D; j++)
        {
            GB.push_back(j);
        }
        GBs.push_back(GB);
    }

    std::vector<int> dev;
    for (int i = N - D; i < N; i++) {
        dev.push_back(i - D);
    }

    for (size_t i = 0; i < dev.size(); i += C) {
        std::vector<int> GB;
        for (size_t j = i; j < i + C && j < dev.size(); j++)
        {
            GB.push_back(dev[j]);
        }
        GBs.push_back(GB);
    }
    // printf("[createGateBlock]: finish\n");
    // printGateBlock(GBs);

    Qureg qureg = createStateVec(N, D, B, T, C, rank);
    printf("[createState]: finish\n");
    if (qureg.rankId == 0) {
        qureg.stateVec[0].real = 1.0f;
    }
    // print machine information
    // printf("rankId: %d, numRanks: %d, numQubitPerRank: %d, numAmpPerRank: %llu\n",
    //         qureg.rankId, qureg.numRanks, qureg.numQubitPerRank, qureg.numAmpPerRank);

    bool* graph = init2DGraph(qureg);
    double *weights = initWeights(qureg);
    // ull *graph = init1DGraph(qureg);

    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            addEdgeTo2DGraph(qureg, graph, weights, i, j, 1.0f);  // add edge that connects node i and node j
            // addEdgeTo1DGraph(qureg, graph, i, j); // add edge that connects node i and node j

    initH(qureg);
    std::vector<int> qubitMap(N);
    for (int i = 0; i < N; i++) {
        qubitMap[i] = i;
    }

    printf("Simulator Start!!\n");

    MEASURET_START;
    for (int p = 0; p < P; p++)
    {
        bool csqsflag = true;
#ifdef USE_AVX
        // rotationCompressionUnweighted_avx(qureg, gamma, graph, p==0);
        rotationCompressionWeighted_avx(qureg, gamma, graph, weights, p==0);
        multiRotateX_avx(qureg, beta, GBs, csqsflag, qubitMap);
#else
        // rotationCompressionUnweighted(qureg, gamma, graph, p == 0);
        rotationCompressionWeighted(qureg, gamma, graph, weights, p==0);
        multiRotateX(qureg, beta, GBs, csqsflag, qubitMap);
#endif
    }
    MEASURET_END;

    if (qureg.rankId == 0) {
        printf("Elapsed time: %f ms\n", diff / 1000.f);
        // printStateVec(qureg, 1);
    }
    destroyStateVec(qureg);
    free2DGraph(graph);
    // free1DGraph(graph);
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
