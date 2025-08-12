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
 
#include "rzzGate.h"

extern double H;

inline void sincosf64(double angle, double* sinRad, double* cosRad) {
    *sinRad = sin(angle);
    *cosRad = cos(angle);
}

void rotationCompressionWeighted(Qureg qureg, double angle, bool *graph, double *weights, bool isFirstLayer)
{
    int numQubits = qureg.numQubitTotal;
    ull numAmps = 1ull << qureg.numQubitPerRank;

    #pragma omp parallel for
    for (ull q = 0; q < numAmps; q++)
    {
        double sum = 0;
        // count number of 1
        for (int i = 0; i < numQubits; i++)
        {
            for (int j = i + 1; j < numQubits; j++)
            {
                int idx = i * numQubits + j;
                if (graph[idx])
                {
                    if (((q >> i) & 0b1) ^ ((q >> j) & 0b1))
                        sum += weights[idx];
                    else
                        sum -= weights[idx];
                }
            }
        }

        double contracted_rzz[2];

        sincosf64(sum * angle / 2.0, &contracted_rzz[1], &contracted_rzz[0]);

        if (isFirstLayer)
        {
            qureg.stateVec[q].real = contracted_rzz[0] * H;
            qureg.stateVec[q].imag = contracted_rzz[1] * H;
        }
        else
        {
            double in[2];
            in[0] = qureg.stateVec[q].real;
            in[1] = qureg.stateVec[q].imag;
            qureg.stateVec[q].real = in[0] * contracted_rzz[0] - in[1] * contracted_rzz[1];
            qureg.stateVec[q].imag = in[0] * contracted_rzz[1] + in[1] * contracted_rzz[0];
        }
    }
}

void rotationCompressionUnweighted(Qureg qureg, double angle, ull *graph, bool isFirstLayer)
{
    int numQubits = qureg.numQubitTotal;
    ull numAmps = 1ull << qureg.numQubitPerRank;

    int numGates = 0;
    for (int i = 0; i < numQubits; i++)
        numGates += __builtin_popcountll(graph[i]);

    #pragma omp parallel for
    for (ull q = 0; q < numAmps; q += 1)
    {
        ull b = q + (qureg.rankId * numAmps);
        int Cp = 0;
        // count number of 1
        for (int i = 0; i < numQubits; i++)
        {
            ull strxor;
            ull msk = -((b >> i) & 1LL);
            strxor = graph[i] & (b ^ msk);
            Cp += __builtin_popcountll(strxor);
        }

        double contracted_rzz[2];
        sincosf64((2 * Cp - numGates) * angle / 2, &contracted_rzz[1], &contracted_rzz[0]);
        if (isFirstLayer)
        {
            qureg.stateVec[q].real = contracted_rzz[0] * H;
            qureg.stateVec[q].imag = contracted_rzz[1] * H;
        }
        else
        {
            double in[2];
            in[0] = qureg.stateVec[q].real;
            in[1] = qureg.stateVec[q].imag;
            qureg.stateVec[q].real = in[0] * contracted_rzz[0] - in[1] * contracted_rzz[1];
            qureg.stateVec[q].imag = in[0] * contracted_rzz[1] + in[1] * contracted_rzz[0];
        }
    }
}