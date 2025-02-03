#include <math.h>
#include <stdio.h>
#include <bitset>

#include "graph.h"
#include "state.h"
#include "rx.h"
#include "rzz.h"
#include "swap.h"

#include "helper_cuda.h"
#include "utils.h"

#define GAMMA 0.2 * M_PI * 2
#define BETA 0.15 * M_PI * 2

int main(int argc, char* argv[]) {

    int P;  // number of rounds
    int N;  // number of qubits
    int C; // chunk size
    // int B;  // number of buffers per device
    // int A;  // number of amps per buffer


    if (argc >= 2) {
        P = atoi(argv[1]);
        N = atoi(argv[2]);
        C = atoi(argv[3]);
        // B = 1 << atoi(argv[3]);
        // A = 1 << atoi(argv[4]);
    } else {
        P = 1;
        N = 30;
        C = 10;
        // B = 1 << 1;
        // A = 1 << 27;
    }

    // printf("%6s, %15s, %22s\n", "qubits", "time (us)", "stateVector[0]");

    State qureg = createState(N);

    ull* graph = init1DGraph(qureg);

    // Fully connected graph
    for (int i=0; i<N; i++)
        for (int j=i+1; j<N; j++)
            addEdgeTo1DGraph(qureg, graph, i, j);  // add edge that connects node i and node j
    // Initialize for Launch control
    initH(qureg);


    // ==========================================================
    // Initialize Subcircuits.
    // ==========================================================
    int chunk_size = C;
    int num_subcir  = 0;
    vector<vector<int>> subcirs;   
    for (int i = 0; i < N; i += chunk_size) {
        num_subcir += 1;
        vector<int> qubitIdx_in_current_subcir;
        for (int j = i; j < i + chunk_size && j < N; ++j) {
            qubitIdx_in_current_subcir.push_back(j);
        }
        subcirs.push_back(qubitIdx_in_current_subcir);
    }


    int numGates = 0;
    for (int i = 0; i < qureg.numQubits; i++) {
        numGates += __builtin_popcountll(graph[i]);
        std::cout << std::bitset<25>(numGates) << std::endl;
    }
    // ==========================================================
    // Start implementation.
    // ==========================================================
    MEASURET_START;

    for (int p=0; p<P; p++) {
        // =======================================================
        // RZZ Gate.
        // =======================================================
        rotationCompressionUnweighted(qureg, GAMMA, graph, p==0);

        // std::cout << "Tot num_subcir = " << num_subcir << endl;
        int swapIn_index = 0;
        for(int i = 0; i < num_subcir; i++) {
            // cout << "num_subcir = " << i << endl; 
            // for (auto &q : subcirs[i])
            //     q %= chunk_size;
            // ===================================================
            // RX Gate.
            // ===================================================
            multiRotateX(qureg, subcirs[0], subcirs[i].size(), BETA);
            // ===================================================
            // Recover state vector.
            // ===================================================
            if( i > 0 ){
                swap_gate_conti(qureg, 0, swapIn_index, subcirs[i].size());
                mapping_after_swap_conti(qureg, graph, 0, swapIn_index, subcirs[i].size());
            }
            // ===================================================
            // Swap in chunk.
            // ===================================================
            if( i < num_subcir-1 ){
                // cout << "swap = " << i << " & " << i+1 << endl;
                swapIn_index += subcirs[i].size();
                swap_gate_conti(qureg, 0, swapIn_index, subcirs[i+1].size());
                mapping_after_swap_conti(qureg, graph, 0, swapIn_index, subcirs[i+1].size());
            }
        }  
    }

    // ==========================================================
    // Synchronize.
    // ==========================================================
    checkCudaErrors(cudaDeviceSynchronize());

    // ==========================================================
    // Debug.
    // ==========================================================
    // double ans = pow(1./sqrt(2), N);
    // for(int i = 0; i < (1 << N); i++){
    //     // cout << i << ":" << qureg.hState[i].real << endl;
    //     if(abs(qureg.hState[i].real - ans) < 1e-9 ){
    //         continue;
    //     }
    //     else {
    //         cout << "Error in index " << i << ". it's val = " << qureg.hState[i].real << endl; return;
    //     }
    // }

    MEASURET_END;

    stateCpyDeviceToHost(qureg);

    printf("%6d, %10.3f, % .6f + % .6fi\n", N, diff / 1000000.0, qureg.hState[0].real, qureg.hState[0].imag);
    fflush(stdout);
    free1DGraph(graph);

    destroyState(qureg); 

    return 0;
}