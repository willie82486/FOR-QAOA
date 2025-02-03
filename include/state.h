#ifndef _STATE_H_
#define _STATE_H_

#include <nccl.h>
#define RANK_PER_NODE 8

typedef double Fp;

typedef unsigned long long ull;
typedef long long ll;

typedef struct  Complex{
    Fp real;
    Fp imag;
} Complex;

typedef struct {
    Complex* dState;
    Complex* dBuf;
    Complex* dTexMem;
    cudaTextureObject_t texObj;
    cudaStream_t compute_stream;
    ncclComm_t comm;
    void * stateHandle;
    void * bufHandle;
    int id;
#if USE_MPI
    int world_rank;
#endif
    Fp* d_graph_1D;
    int *d_subcirs;
    int *d_table;
} GPUState;


typedef struct State
{
    Complex* hState;
    int numDevice;
    int numQubits;
    int numQubitsPerDevice;
    int numBufsPerDevice;
    int numQubitsPerBuffer;
    ull numAmpsPerDevice;
    ull stateSizePerDevice;
    ull numAmpsPerBuf;
    ull bufSize;
    GPUState *gpus;
} State;

// State createState(int numQubits, int numBufsPerDevice, ull numAmpsPerBuf);
State createMPIState(int& N, int& D, int& B, int& C, int& world_rank, int& world_size);
State createState(int N, int D, int B, int C);
void destroyState(const State& state);

void stateCpyDeviceToHost(const State& state);
void stateCpyHostToDevice(const State& state);

void stateInitDeviceState(const State& qureg);
void ncclWarnup(State &qureg);

void prepare(State &qureg,int C);
#endif