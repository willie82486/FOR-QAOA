#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <vector>

#include <nccl.h>
#include "state.h"
#include "helper_cuda.hpp"





State createState(int N, int D, int B,int C) {
    State qureg;
    qureg.numQubits = N;

    qureg.numDevice = 1 << D;
    qureg.numQubitsPerDevice = N - D;
    qureg.numAmpsPerDevice = 1ull << qureg.numQubitsPerDevice;
    qureg.stateSizePerDevice = qureg.numAmpsPerDevice * sizeof(Complex);

    qureg.numQubitsPerBuffer = B;
    qureg.numAmpsPerBuf = 1ull << B;
    qureg.bufSize = qureg.numAmpsPerBuf * sizeof(Complex);
    qureg.gpus = (GPUState*) malloc(qureg.numDevice * sizeof(GPUState));
    memset(qureg.gpus, 0, qureg.numDevice * sizeof(GPUState));
    
    ncclComm_t* comms = (ncclComm_t*)malloc(qureg.numDevice * sizeof(ncclComm_t));

    if (qureg.numDevice != 1 && qureg.bufSize)
        checkCudaErrors(ncclCommInitAll(comms, qureg.numDevice, NULL));
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        // NOTE: it will cause an error if peer access has already been enabled,
        //       so we don't wrap it with checkCudaErrors()
        for (int remDev = 0; remDev < qureg.numDevice; remDev++) {
            if (remDev != dev)
                cudaDeviceEnablePeerAccess(remDev, 0);
        }
        GPUState &gState = qureg.gpus[dev];
        gState.id = dev;
        if (qureg.numDevice != 1 && qureg.bufSize )
        {
            gState.comm = comms[dev];
            checkCudaErrors(ncclMemAlloc((void **)&gState.dState, qureg.stateSizePerDevice));
            checkCudaErrors(ncclCommRegister(gState.comm, gState.dState, qureg.stateSizePerDevice, &gState.stateHandle));
            checkCudaErrors(ncclMemAlloc((void **)&gState.dBuf, qureg.bufSize));
            checkCudaErrors(ncclCommRegister(gState.comm, gState.dBuf, qureg.bufSize, &gState.bufHandle));
        }
        else
        {
            checkCudaErrors(cudaMalloc(&gState.dState, qureg.stateSizePerDevice));
        }
        checkCudaErrors(cudaStreamCreateWithFlags(&gState.compute_stream, cudaStreamNonBlocking));
    }

    // if (qureg.numDevice != 1 && qureg.bufSize) {
    //     qureg.dBufs = (Complex **) malloc(qureg.numDevice * sizeof(Complex *));
    //     for (int dev = 0; dev < qureg.numDevice; dev++) {
    //         checkCudaErrors(cudaMalloc(&qureg.dBufs[dev], qureg.bufSize));
    //     }
    // }

    // ull stateSize = qureg.numDevice * qureg.stateSizePerDevice;
    ull stateSize = 1ull << qureg.numQubits;

    qureg.hState = (Complex*) calloc(stateSize, sizeof(Complex));
    assert(qureg.hState);

    // memset(qureg.hState, 0, stateSize);
    qureg.hState[0].real = (Fp)1;
    ncclWarnup(qureg);
    stateInitDeviceState(qureg);
    prepare(qureg, C);
    return qureg;
}

void prepare(State &qureg, int C){
    //rzz
    size_t size = (qureg.numQubits * (qureg.numQubits-1))/2 * sizeof(Fp);
	ull numSwapPairs = (1ull<<C) * ((1ull<<C)-1) / 2;
    ull swap_table_size = numSwapPairs * 2 * sizeof(int);
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        //rzz
        checkCudaErrors(cudaMallocAsync(&qureg.gpus[dev].d_graph_1D, size, qureg.gpus[dev].compute_stream));
        //rx
        checkCudaErrors(cudaMallocAsync(&qureg.gpus[dev].d_subcirs, C * sizeof(int), qureg.gpus[dev].compute_stream));
        //swap
        checkCudaErrors(cudaMallocAsync(&qureg.gpus[dev].d_table, swap_table_size, qureg.gpus[dev].compute_stream));

    }
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    //rx
}

void stateInitDeviceState(const State& qureg) {   
    Complex one = {1, 0};
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        checkCudaErrors(cudaMemset(qureg.gpus[dev].dState, 0, qureg.stateSizePerDevice));
    }
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMemcpyAsync(qureg.gpus[0].dState, &one, sizeof(one), cudaMemcpyHostToDevice));
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}


void ncclWarnup(State &qureg) {
    
    const int &N = qureg.numQubits;
    const int &D = qureg.numQubits - qureg.numQubitsPerDevice;
    const int &B = qureg.numQubitsPerBuffer;
    const int &csqsSize = D;

    const int numGroups = 1 << (D - csqsSize);
    const int numMemberInGroup = 1 << csqsSize;

    std::vector<std::vector<int>> devlist(numGroups, std::vector<int>(numMemberInGroup));

    for (int mbr = 0; mbr < numMemberInGroup; mbr++) {
        devlist[0][mbr] = mbr;
    }

    for (ull off = 0; off < (1ull << (N - D - csqsSize)); off += (1 << (B - csqsSize)))
    {
        for (int grp = 0; grp < (1 << (D - csqsSize)); grp++)
        {
            checkCudaErrors(ncclGroupStart());
            for (int a = 0; a < (1 << csqsSize); a++)
            {
                int devA = devlist[grp][a];
                // checkCudaErrors(ncclGroupStart());
                for (int b = 0; b < (1 << csqsSize); b++)
                {
                    if (a == b)
                        continue;
                    int devB = devlist[grp][b];
                    int off_sv = b * (1ull << (N - D - csqsSize)) + off;
                    int off_bf = b * (1ull << (B - csqsSize));
                    checkCudaErrors(ncclSend(qureg.gpus[devA].dState + off_sv, (1 << (B - csqsSize)) * sizeof(Complex), ncclChar, devB, qureg.gpus[devA].comm, qureg.gpus[devA].compute_stream));
                    checkCudaErrors(ncclRecv(qureg.gpus[devA].dBuf + off_bf, (1 << (B - csqsSize)) * sizeof(Complex), ncclChar, devB, qureg.gpus[devA].comm, qureg.gpus[devA].compute_stream));
                }
                // checkCudaErrors(ncclGroupEnd());
            }
            checkCudaErrors(ncclGroupEnd());
        }
        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            checkCudaErrors(cudaDeviceSynchronize());
        } 


        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaSetDevice(dev));
            checkCudaErrors(cudaDeviceSynchronize());
        }  
        for (int grp = 0; grp < (1 << (D - csqsSize)); grp++)
        {
            for (int a = 0; a < (1 << csqsSize); a++)
            {
                int devA = devlist[grp][a];
                // checkCudaErrors(ncclGroupStart());
                for (int b = 0; b < (1 << csqsSize); b++)
                {
                    if (a == b)
                        continue;
                    int off_sv = b * (1ull << (N - D - csqsSize)) + off;
                    int off_bf = b * (1ull << (B - csqsSize));
                    checkCudaErrors(cudaMemcpyAsync(qureg.gpus[devA].dState + off_sv, qureg.gpus[devA].dBuf + off_bf, (1 << (B - csqsSize)) * sizeof(Complex), cudaMemcpyDeviceToDevice, qureg.gpus[devA].compute_stream));
                }
                // checkCudaErrors(ncclGroupEnd());
            }
        }

        break;
    }
}

void stateCpyDeviceToHost(const State& qureg) {
    size_t offset = 0;
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        checkCudaErrors(cudaMemcpy(qureg.hState + offset, qureg.gpus[dev].dState, qureg.stateSizePerDevice, cudaMemcpyDeviceToHost));
        offset += qureg.numAmpsPerDevice;
    }
}

void stateCpyHostToDevice(const State& qureg) {
    size_t offset = 0;
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        checkCudaErrors(cudaMemcpy(qureg.gpus[dev].dState, qureg.hState + offset, qureg.stateSizePerDevice, cudaMemcpyHostToDevice));
        offset += qureg.numAmpsPerDevice;
    }
}

void destroyState(const State& qureg) {
    for (int dev = 0; dev < qureg.numDevice; dev++) {
        checkCudaErrors(cudaSetDevice(dev));
        checkCudaErrors(cudaFree(qureg.gpus[dev].dState));
    }
    if (qureg.numDevice != 1 && qureg.bufSize) {
        for (int dev = 0; dev < qureg.numDevice; dev++) {
            checkCudaErrors(cudaFree(qureg.gpus[dev].dBuf));
        }
        free(qureg.gpus);
    }
    free(qureg.hState);
}