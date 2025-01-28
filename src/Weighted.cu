#include <math.h>
#include <stdio.h>

#include "graph.h"
#include "state.h"
#include "rx.h"
#include "rzz.h"
#include "swap.h"

#include "helper_cuda.h"
#include "utils.h"

#if USE_MPI
#include <mpi.h>
#endif

#define GAMMA M_PI_4
#define BETA M_PI_4

int main(int argc, char* argv[])
{
#if USE_MPI
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the local rank on the node 
    int local_rank = 0;
    if (const char* val = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
        local_rank = std::atoi(val);
    }
#endif
    int P;  // number of rounds
    int N;  // number of qubits
    int D;  // number of devices
    int C;  // chunk size
    int B;  // number of amps per buffer
    // printf("argc = %d", argc);
    if (argc >= 2) {
        P = atoi(argv[1]);
        N = atoi(argv[2]);
        D = atoi(argv[3]);
        C = atoi(argv[4]);
        B = atoi(argv[5]);
    } 
    else {
        P = 1;
        N = 21;
        D = 0;
        C = 10;
        B = 19;
    }
    // cout << P << " ";
    // cout << N << " ";
    // cout << C << " ";
    // cout << A << endl;
    // printf("%6s, %15s, %22s\n", "qubits", "time (us)", "stateVector[0]");

    // for (int num_qubits = 2; num_qubits <= 30; num_qubits++) {
    cout << P << "," << N << "," << D << "," << C << "," << B;

        State qureg = createState(N, D, B, C);
#if 1
        Fp* graph = init2DGraph(qureg);
        
        // fully connected graph
        for (int i=0; i<N; i++)
            for (int j=i+1; j<N; j++)
                addEdgeTo2DGraph(qureg, graph, i, j, 1.f);    // add edge that connects node i and node j


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

        // vector<vector<int>> subcirs = {{0, }};
        // num_subcir += 1;

        // ==========================================================
        // Start implementation.
        // ==========================================================
        MEASURET_START;
        
        for (int p=0; p<P; p++) {
            // =======================================================
            // RZZ Gate.
            // =======================================================
            rotationCompressionWeighted(qureg, GAMMA, graph, p==0);
            
            // std::cout << "Tot num_subcir = " << num_subcir << endl;
            int swapIn_index = 0;
            for(int i = 0; i < num_subcir; i++) {
                // cout << "num_subcir = " << i << endl; 
                // for (auto &q : subcirs[i])
                //     q %= chunk_size;
                // ===================================================
                // RX Gate.
                // ===================================================
                // printf("i = %d \n", i);
                multiRotateX(qureg, subcirs[0], subcirs[i].size(), BETA);
                // ===================================================
                // Recovering.
                // ===================================================
                // if( i > 0 ){
                //     swap_gate_conti(qureg, 0, swapIn_index, subcirs[i].size());
                //     // mapping_after_swap_conti(qureg, graph, 0, swapIn_index, subcirs[i].size());
                // }
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
        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            checkCudaErrors(cudaDeviceSynchronize());
        }  
        MEASURET_END;


        // stateCpyDeviceToHost(qureg);
        // ofstream file;
        // file.open("dmp.txt", ios::binary | ios::out);
        // file.write((char*)qureg.hState, qureg.numDevice * qureg.stateSizePerDevice);
        // file.close();

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



// stateCpyDeviceToHost(qureg); //unfinished

        // printf("%6d, %15lu, % .6f + % .6fi\n", N, diff, qureg.hState[0].real, qureg.hState[0].imag);
        printf(",%10.3f\n", diff / 1000000.0);
        fflush(stdout);
        free2DGraph(graph);
        // cout << endl << endl << "=============================================" << endl;

// destroyState(qureg);        //unfinished
#endif
    // }

    return 0;
}