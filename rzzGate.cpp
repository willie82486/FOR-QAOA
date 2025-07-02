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