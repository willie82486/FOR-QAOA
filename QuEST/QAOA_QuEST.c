#include <QuEST.h>
#include <stdio.h>

#include "../include/utils.h"

#define GAMMA 0.2 * M_PI * 2
#define BETA 0.15 * M_PI * 2

int main(int argc, char *argv[]) {

    int P;  // number of rounds

    if (argc >= 2) {
        P = atoi(argv[1]);
    } else {
        P = 1;
    }

    printf("%6s, %15s, %22s\n", "qubits", "time (us)", "stateVector[0]");

    for (int num_qubits = 26; num_qubits <= 27; num_qubits++) {

        QuESTEnv env = createQuESTEnv();
        Qureg qubits = createQureg(num_qubits, env);

        MEASURET_START;

        for (int i=0; i<num_qubits; i++)
           hadamard(qubits, i);

        for (int p=0; p<P; p++) {

            // Fully connected graph
            for (int i=0; i<num_qubits; i++) {
               for (int j=i+1; j<num_qubits; j++) {
                   int targets[2] = {i, j};
                   multiRotateZ(qubits, targets, 2, GAMMA);
               }
            }

            for (int i=0; i<num_qubits; i++) {
               rotateX(qubits, i, BETA);
            }
        }

        syncQuESTEnv(env);

        MEASURET_END;

        // copyStateFromGPU(qubits);

        printf("%6d, %10.3f, % .6f + % .6fi\n", num_qubits, diff / 1000000.0, qubits.stateVec.real[0], qubits.stateVec.imag[0]);
        fflush(stdout);

        // unload QuEST
        destroyQureg(qubits, env); 
        destroyQuESTEnv(env);
    }

    return 0;
}